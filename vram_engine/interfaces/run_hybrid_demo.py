"""BM25 + isomorphism(스펙트럴) 하이브리드 검색 + Agent 전체 워크플로우 +
실시간 대시보드 데모.

사용법:
    python -m vram_engine.interfaces.run_hybrid_demo            # 콘솔
    python -m vram_engine.interfaces.run_hybrid_demo --gui       # 대시보드

호환성 노트:
    dashboard.py의 retriever= 파라미터는 `.index()`, `.query()`,
    `.documents`, `.doc_adj`, `.prev_entropy`를 갖는 객체를 기대한다
    (구 HybridBM25SpectralRetriever의 인터페이스). 이번 리팩터링으로
    검색 로직이 DocumentStore/GraphStore/BM25Retriever/SpectralRetriever/
    HybridRetriever/Verifier/CircuitBreaker/Agent/Synthesizer 여러
    클래스로 쪼개졌지만, dashboard.py를 고치는 대신 아래 
    LegacyRetrieverAdapter가 그 옛 인터페이스를 그대로 흉내 내서
    내부적으로 새 구조(Agent)를 호출하도록 했다. 기존 API를 불필요하게
    깨뜨리지 않기 위한 어댑터 패턴이다.
"""
import sys
from typing import List

from vram_engine.agent.agent import RetrievalAgent
from vram_engine.agent.planner import Planner
from vram_engine.core.document_store import DocumentStore
from vram_engine.core.graph_store import GraphStore
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
    "강화학습은 보상을 최대화하는 정책을 학습하는 방법이다",
    "양자 컴퓨터는 큐비트의 중첩과 얽힘을 이용해 계산한다",
]

DEMO_QUERIES = [
    "양자 확률 붕괴",
    "김치 요리법",
    "신경망 확률 정규화",
    "보상 최대화 정책",
    "큐비트 중첩",
]


class LegacyRetrieverAdapter:
    """구 HybridBM25SpectralRetriever와 동일한 `.index()`/`.query()` 인터페이스.
    내부는 새 계층 구조(Agent)를 그대로 사용한다."""

    def __init__(
        self,
        iso_engine: CrossDomainIsomorphismEngine,
        verifier: AntiHallucinationVerifier,
        circuit_breaker: SelfHealingCircuitBreaker,
        alpha: float = 0.5,
    ):
        self.document_store = DocumentStore()
        self.graph_store: GraphStore = None
        self.bm25 = BM25Retriever()
        self.spectral = SpectralRetriever(iso_engine)
        self.hybrid = HybridRetriever(self.bm25, self.spectral, alpha=alpha)
        self.verifier = verifier
        self.circuit_breaker = circuit_breaker
        self.synthesizer = Synthesizer()
        self._alpha = alpha
        self.agent: RetrievalAgent = None

    @property
    def documents(self) -> List[str]:
        return [d.text for d in self.document_store.all()]

    @property
    def doc_adj(self):
        return self.graph_store.current() if self.graph_store else None

    @property
    def prev_entropy(self) -> float:
        return self.agent._prev_entropy if self.agent else 0.0

    def index(self, documents: List[str]) -> None:
        self.document_store.add_many(documents)
        adjacency = build_similarity_graph(self.document_store.all())
        self.graph_store = GraphStore(adjacency)
        self.bm25.index(self.document_store.all())

        # confidence_threshold=0.0 -> 항상 첫 결과를 채택 (재검색 없음).
        # 구 hybrid_retriever.query()의 단일 패스 동작과 동일하게 만들어
        # 기존 데모 출력이 그대로 재현되도록 한다.
        planner = Planner(base_alpha=self._alpha, base_top_k=len(self.document_store), confidence_threshold=0.0)
        self.agent = RetrievalAgent(
            hybrid_retriever=self.hybrid,
            verifier=self.verifier,
            circuit_breaker=self.circuit_breaker,
            graph_store=self.graph_store,
            document_store=self.document_store,
            planner=planner,
            synthesizer=self.synthesizer,
            max_retries=0,
        )

    def query(self, query_text: str, top_k: int = 5) -> dict:
        final_answer, _trace = self.agent.run(query_text)
        verif_by_id = {v.doc_id: v for v in final_answer.verifications}

        results = []
        for c in final_answer.top_documents[:top_k]:
            v = verif_by_id.get(c.doc_id)
            results.append({
                "document": self.document_store.get(c.doc_id).text,
                "bm25_score": round(c.bm25_score, 4),
                "spectral_score": round(c.spectral_score, 6),
                "hybrid_score": round(c.hybrid_score, 4),
                "verified": v.verified if v else True,
            })

        return {
            "results": results,
            "engine_status": final_answer.engine_status.status,
            "entropy": round(final_answer.engine_status.entropy, 4),
        }


def build_engine(alpha: float = 0.5):
    iso_engine = CrossDomainIsomorphismEngine()
    verifier = AntiHallucinationVerifier()
    circuit_breaker = SelfHealingCircuitBreaker()
    retriever = LegacyRetrieverAdapter(iso_engine, verifier, circuit_breaker, alpha=alpha)
    retriever.index(CORPUS)
    return iso_engine, verifier, circuit_breaker, retriever


def run_console_demo(retriever: LegacyRetrieverAdapter):
    for q in DEMO_QUERIES:
        res = retriever.query(q, top_k=3)
        print("=" * 60)
        print(f"질의: {q}  | 엔진 상태: {res['engine_status']}  | 엔트로피: {res['entropy']}")
        for r in res["results"]:
            print(
                f"  hybrid={r['hybrid_score']:.3f}  bm25={r['bm25_score']:.3f}  "
                f"spectral={r['spectral_score']:.6f}  verified={r['verified']}  "
                f":: {r['document']}"
            )


def run_gui_demo(iso_engine, verifier, circuit_breaker, retriever):
    import tkinter as tk
    from vram_engine.interfaces.dashboard import AutonomousGUIDashboard

    root = tk.Tk()
    AutonomousGUIDashboard(
        root, iso_engine, verifier, circuit_breaker,
        retriever=retriever, demo_queries=DEMO_QUERIES,
    )
    root.mainloop()


if __name__ == "__main__":
    iso_engine, verifier, circuit_breaker, retriever = build_engine()

    if "--gui" in sys.argv:
        run_gui_demo(iso_engine, verifier, circuit_breaker, retriever)
    else:
        run_console_demo(retriever)
