from vram_engine.agent.agent import RetrievalAgent, evaluate_confidence
from vram_engine.agent.planner import Planner
from vram_engine.agent.strategy import IncreaseSpectralWeightStrategy
from vram_engine.core.document_store import DocumentStore
from vram_engine.core.graph_store import GraphStore
from vram_engine.core.types import AgentStepResult, EngineStatus, RetrievalCandidate, RetrievalPlan
from vram_engine.graph.builder import build_similarity_graph
from vram_engine.graph.isomorphism import CrossDomainIsomorphismEngine
from vram_engine.graph.verifier import AntiHallucinationVerifier
from vram_engine.retrieval.bm25 import BM25Retriever
from vram_engine.retrieval.hybrid import HybridRetriever
from vram_engine.retrieval.spectral import SpectralRetriever
from vram_engine.safety.circuit_breaker import SelfHealingCircuitBreaker
from vram_engine.synthesis.synthesizer import Synthesizer

CORPUS = [
    "양자역학에서 파동함수의 붕괴는 관측 시점에 확률적으로 결정된다",
    "소프트맥스 함수는 신경망 출력을 확률 분포로 정규화한다",
    "김치찌개는 돼지고기와 신 김치를 넣어 끓이는 한국 음식이다",
]


def _build_agent(max_retries=0, confidence_threshold=0.0):
    ds = DocumentStore()
    ds.add_many(CORPUS)
    gs = GraphStore(build_similarity_graph(ds.all()))
    iso = CrossDomainIsomorphismEngine()
    verifier = AntiHallucinationVerifier()
    cb = SelfHealingCircuitBreaker()
    bm25 = BM25Retriever()
    bm25.index(ds.all())
    spectral = SpectralRetriever(iso)
    hybrid = HybridRetriever(bm25, spectral, alpha=0.5)
    planner = Planner(base_alpha=0.5, base_top_k=len(ds), confidence_threshold=confidence_threshold)
    synth = Synthesizer()
    agent = RetrievalAgent(hybrid, verifier, cb, gs, ds, planner, synth, max_retries=max_retries)
    return agent, ds


def test_first_query_returns_normal_results_regardless_of_circuit_breaker():
    """핵심 회귀 테스트: 첫 질의에서 그래프에 노드가 추가되며 발생할 수 있는
    엔트로피 변화와 무관하게, 검색 결과 자체는 항상 정상적으로 반환되어야
    한다 (circuit breaker가 무엇을 판정하든 top_documents가 비어있으면
    안 된다)."""
    agent, ds = _build_agent()
    final, _trace = agent.run("양자 확률 붕괴")
    assert len(final.top_documents) == len(CORPUS)
    assert final.engine_status.status in ("NORMAL", "WARNING_PURGE", "CRITICAL_ROLLBACK")
    assert ds.get(final.top_documents[0].doc_id).text == CORPUS[0]


def test_first_query_does_not_spuriously_trigger_critical_rollback():
    """엔트로피 비교 기준을 (N+1)-노드 중립 그래프로 맞춘 덕분에, 정상적인
    첫 질의에서 구조적 노드 추가만으로 CRITICAL_ROLLBACK이 발생하지
    않아야 한다."""
    agent, ds = _build_agent()
    final, _trace = agent.run("양자 확률 붕괴")
    assert final.engine_status.status == "NORMAL"


def test_critical_rollback_restores_graph_store_to_previous_state():
    """circuit breaker가 실제로 CRITICAL_ROLLBACK을 발생시키면, GraphStore가
    이전 안정 상태로 복원되는지 확인 (매우 낮은 entropy_spike_limit으로
    강제 유발)."""
    ds = DocumentStore()
    ds.add_many(CORPUS)
    gs = GraphStore(build_similarity_graph(ds.all()))
    iso = CrossDomainIsomorphismEngine()
    verifier = AntiHallucinationVerifier()
    cb = SelfHealingCircuitBreaker(entropy_spike_limit=1e-9, max_hallucination_streak=999)
    bm25 = BM25Retriever()
    bm25.index(ds.all())
    spectral = SpectralRetriever(iso)
    hybrid = HybridRetriever(bm25, spectral, alpha=0.5)
    planner = Planner(base_alpha=0.5, base_top_k=len(ds), confidence_threshold=0.0)
    synth = Synthesizer()

    original_graph = gs.current().clone()
    agent = RetrievalAgent(hybrid, verifier, cb, gs, ds, planner, synth, max_retries=0)
    final, _trace = agent.run("양자 확률 붕괴")

    assert final.engine_status.status == "CRITICAL_ROLLBACK"
    # 그래프 스토어가 복원되어 있어야 한다 (초기 스냅샷 기준)
    assert gs.current().shape == original_graph.shape


def test_agent_orchestrates_without_mutating_hybrid_retriever_alpha_permanently():
    agent, ds = _build_agent()
    original_alpha = agent.hybrid_retriever.alpha
    agent.run("김치 요리법")
    # 재검색이 없었으므로(max_retries=0) alpha가 최종적으로 plan의 alpha와 같아야 함
    assert agent.hybrid_retriever.alpha == original_alpha


def test_evaluate_confidence_uses_top_score_and_verified_ratio():
    step = AgentStepResult(
        plan=RetrievalPlan(query_text="q", alpha=0.5, top_k=1),
        candidates=[RetrievalCandidate(doc_id=0, bm25_score=1.0, spectral_score=0.0, hybrid_score=0.8)],
        verifications=[],
        confidence=0.0,
        engine_status=EngineStatus(status="NORMAL", entropy=0.1),
    )
    assert evaluate_confidence(step) == 0.8  # verifications 없으면 ratio=1.0


def test_retry_strategy_lowers_alpha():
    plan = RetrievalPlan(query_text="q", alpha=0.5, top_k=3)
    strategy = IncreaseSpectralWeightStrategy(step=0.2)
    new_plan = strategy.apply(plan)
    assert new_plan.alpha == 0.3
    assert new_plan.query_text == "q"


def test_agent_can_retry_when_confidence_threshold_is_high():
    """confidence_threshold를 매우 높게 설정하면 max_retries만큼 재검색이
    일어나는지 확인 (각 단계가 독립적으로 동작함을 보여주는 테스트)."""
    agent, ds = _build_agent(max_retries=2, confidence_threshold=999.0)
    final, trace = agent.run("양자 확률 붕괴")
    assert len(trace) == 3  # 최초 1회 + 재시도 2회
    assert final.retries == 2
