from vram_engine.core.document_store import DocumentStore
from vram_engine.core.graph_store import GraphStore
from vram_engine.graph.builder import build_similarity_graph
from vram_engine.graph.isomorphism import CrossDomainIsomorphismEngine
from vram_engine.retrieval.bm25 import BM25Retriever
from vram_engine.retrieval.hybrid import HybridRetriever
from vram_engine.retrieval.spectral import SpectralRetriever


def _build(alpha=0.5):
    ds = DocumentStore()
    ds.add_many([
        "quantum wave function collapse probability",
        "softmax neural network normalization",
        "kimchi stew korean food recipe",
    ])
    gs = GraphStore(build_similarity_graph(ds.all()))
    iso = CrossDomainIsomorphismEngine()
    bm25 = BM25Retriever()
    bm25.index(ds.all())
    spectral = SpectralRetriever(iso)
    hybrid = HybridRetriever(bm25, spectral, alpha=alpha)
    return ds, gs, hybrid


def test_hybrid_retrieve_ranks_matching_document_first():
    ds, gs, hybrid = _build()
    candidates = hybrid.retrieve("quantum probability collapse", ds, gs, top_k=3)
    assert candidates[0].doc_id == 0


def test_hybrid_retrieve_returns_requested_top_k():
    ds, gs, hybrid = _build()
    candidates = hybrid.retrieve("kimchi recipe", ds, gs, top_k=2)
    assert len(candidates) == 2


def test_hybrid_retriever_has_no_verifier_or_circuit_breaker_dependency():
    """책임 분리 회귀 테스트: HybridRetriever는 verifier/circuit_breaker
    모듈을 import하거나 생성자 파라미터로 갖지 않아야 한다."""
    import inspect

    from vram_engine.retrieval.hybrid import HybridRetriever
    from vram_engine.retrieval import hybrid as hybrid_module

    assert "vram_engine.graph.verifier" not in dir(hybrid_module)
    assert not hasattr(hybrid_module, "AntiHallucinationVerifier")
    assert not hasattr(hybrid_module, "SelfHealingCircuitBreaker")

    sig_params = list(inspect.signature(HybridRetriever.__init__).parameters)
    assert "verifier" not in sig_params
    assert "circuit_breaker" not in sig_params
