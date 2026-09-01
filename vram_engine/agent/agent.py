"""전체 워크플로우를 조율하는 Agent.

    Query
     -> Plan (Planner)
     -> Retrieve (HybridRetriever)
     -> Verify (AntiHallucinationVerifier)
     -> Evaluate confidence
     -> 충분하면 Synthesize
     -> 부족하면 Strategy로 새 Plan -> 재검색

circuit breaker는 Agent가 호출하지만, GraphStore를 직접 조작하지 않는다:
circuit_breaker.inspect_and_heal()은 텐서를 돌려줄 뿐이고, 그걸
graph_store.replace()로 실제 반영하는 건 Agent다. 즉 GraphStore
쪽에서 보면 Agent라는 하나의 클라이언트가 read(current)/write(replace)
할 뿐, circuit_breaker가 GraphStore의 존재 자체를 모른다.

첫 질의에서의 entropy spike 문제:
  질의는 매번 그래프에 "새 노드"로 임시 증강되어 (N+1)x(N+1) 그래프의
  엔트로피가 계산된다. 그런데 색인 직후의 초기 baseline 엔트로피는
  질의 노드가 없는 NxN 코퍼스 그래프에서 계산된 값이었다. 첫 질의의
  (N+1)x(N+1) 엔트로피를 이 NxN 기준값과 비교하면, "질의 내용이
  이상해서"가 아니라 "그래프에 노드가 하나 늘어서" 생기는 구조적
  엔트로피 증가가 실제 이상 신호와 뒤섞여 버린다.
  이를 임계값을 조정해서 숨기지 않고, 애초에 비교 기준 자체를
  (N+1)x(N+1) 크기로 맞춰서 만든다 (질의 엣지가 전부 0인 "중립" 증강
  그래프의 엔트로피를 초기 baseline으로 사용). circuit_breaker.py의
  판정 알고리즘 자체는 전혀 건드리지 않는다 — 비교 대상의 스케일만
  맞춘 것이다.
"""
from threading import RLock
from typing import List, Optional, Tuple

from vram_engine.agent.planner import Planner
from vram_engine.agent.strategy import IncreaseSpectralWeightStrategy, RetrievalStrategy
from vram_engine.core.document_store import DocumentStore, hybrid_tokenize
from vram_engine.core.graph_store import GraphStore
from vram_engine.core.types import AgentStepResult, EngineStatus, FinalAnswer, RetrievalPlan
from vram_engine.graph.builder import build_query_edge_weights
from vram_engine.graph.verifier import AntiHallucinationVerifier
from vram_engine.retrieval.hybrid import HybridRetriever
from vram_engine.safety.circuit_breaker import SelfHealingCircuitBreaker
from vram_engine.synthesis.synthesizer import Synthesizer


def evaluate_confidence(step: AgentStepResult) -> float:
    """최상위 후보의 hybrid_score와 검증 통과 비율을 곱해 confidence를 낸다.

    독립적으로 테스트 가능하도록 순수 함수로 분리했다.
    """
    if not step.candidates:
        return 0.0
    top_score = step.candidates[0].hybrid_score
    if step.verifications:
        verified_ratio = sum(1 for v in step.verifications if v.verified) / len(step.verifications)
    else:
        verified_ratio = 1.0
    return top_score * verified_ratio


class RetrievalAgent:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        verifier: AntiHallucinationVerifier,
        circuit_breaker: SelfHealingCircuitBreaker,
        graph_store: GraphStore,
        document_store: DocumentStore,
        planner: Planner,
        synthesizer: Synthesizer,
        default_strategy: Optional[RetrievalStrategy] = None,
        max_retries: int = 1,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.verifier = verifier
        self.circuit_breaker = circuit_breaker
        self.graph_store = graph_store
        self.document_store = document_store
        self.planner = planner
        self.synthesizer = synthesizer
        self.default_strategy = default_strategy or IncreaseSpectralWeightStrategy()
        self.max_retries = max_retries
        self._run_lock = RLock()

        self._last_entropy = self._neutral_entropy()
        circuit_breaker.update_snapshot(graph_store.current(), self._last_entropy)

    @property
    def last_entropy(self) -> float:
        """Most recently completed request entropy for monitoring clients."""
        return self._last_entropy

    def run(self, query_text: str) -> Tuple[FinalAnswer, List[AgentStepResult]]:
        # Agent, circuit breaker, and graph store are intentionally shared by the
        # dashboard. Serialising one complete run keeps that shared safety state
        # coherent while keeping per-request entropy local.
        with self._run_lock:
            prev_entropy = self._neutral_entropy()
            self.circuit_breaker.reset_transient_state()
            self.circuit_breaker.update_snapshot(self.graph_store.current(), prev_entropy)
            plan = self.planner.initial_plan(query_text)
            trace: List[AgentStepResult] = []
            retries = 0

            while True:
                step, prev_entropy = self._run_step(plan, prev_entropy)
                trace.append(step)
                if not self.planner.should_retry(step.confidence, retries, self.max_retries):
                    break
                retries += 1
                plan = self.planner.next_plan(plan, self.default_strategy)

            final_step = trace[-1]
            self._last_entropy = final_step.engine_status.entropy
            final_answer = self.synthesizer.synthesize(
                query_text=query_text,
                candidates=final_step.candidates,
                verifications=final_step.verifications,
                engine_status=final_step.engine_status,
                retries=retries,
            )
            return final_answer, trace

    def _run_step(self, plan: RetrievalPlan, prev_entropy: float) -> Tuple[AgentStepResult, float]:
        candidates = self.hybrid_retriever.retrieve(
            plan.query_text, self.document_store, self.graph_store, top_k=plan.top_k, alpha=plan.alpha
        )

        n = len(self.document_store)
        documents = self.document_store.all()

        # 1. 검증: 질의-문서 엣지가 없는 기준 그래프 위에서 각 후보를 검증
        base_for_check = self.graph_store.augmented_with([0.0] * n)
        candidate_ids = [c.doc_id for c in candidates]
        verifications = self.verifier.verify_candidates(base_for_check, n, candidate_ids)

        # 2. 실제 유사도가 반영된 증강 그래프의 엔트로피로 circuit breaker 판단
        query_tokens = hybrid_tokenize(plan.query_text)
        edge_weights = build_query_edge_weights(query_tokens, documents)
        aug_full = self.graph_store.augmented_with(edge_weights)
        current_entropy = float(self.verifier.calc_entropy(aug_full).item())

        any_hallucination = not all(v.verified for v in verifications)
        healed_state, status = self.circuit_breaker.inspect_and_heal(
            current_entropy, prev_entropy, any_hallucination, aug_full
        )

        if status == "CRITICAL_ROLLBACK":
            # 코퍼스 그래프만 마지막 안정 상태로 되돌린다. GraphStore의
            # 크기(NxN)에 맞춰 잘라서 반영 — circuit_breaker는 이 사실을
            # 모른 채 텐서만 돌려줬을 뿐이다.
            self.graph_store.replace(healed_state[:n, :n].clone())

        engine_status = EngineStatus(status=status, entropy=current_entropy)
        step = AgentStepResult(
            plan=plan,
            candidates=candidates,
            verifications=verifications,
            confidence=0.0,
            engine_status=engine_status,
        )
        step.confidence = evaluate_confidence(step)
        return step, current_entropy

    def _neutral_entropy(self) -> float:
        n = len(self.document_store)
        neutral_aug = self.graph_store.augmented_with([0.0] * n)
        return float(self.verifier.calc_entropy(neutral_aug).item())
