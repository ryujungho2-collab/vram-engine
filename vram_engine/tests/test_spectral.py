import torch

from vram_engine.core.document_store import DocumentStore
from vram_engine.core.graph_store import GraphStore
from vram_engine.graph.builder import build_similarity_graph
from vram_engine.graph.isomorphism import CrossDomainIsomorphismEngine
from vram_engine.retrieval.hybrid import minmax_normalize
from vram_engine.retrieval.spectral import SpectralRetriever


def test_spectral_score_is_bounded_and_nonnegative():
    """회귀 테스트: 고유벡터 부호 모호성 때문에 유사도가 음수/랜덤하게
    나오면 안 된다 — abs()를 취하므로 항상 [0, 1] 범위여야 한다."""
    ds = DocumentStore()
    ds.add_many([
        "quantum wave function collapse",
        "softmax neural network probability",
        "kimchi stew recipe korean food",
    ])
    gs = GraphStore(build_similarity_graph(ds.all()))
    iso = CrossDomainIsomorphismEngine()
    spectral = SpectralRetriever(iso)

    scores = spectral.score("quantum probability collapse", ds.all(), gs)
    for s in scores:
        assert -1e-6 <= s <= 1.0 + 1e-6


def test_noise_floor_prevents_amplifying_numerical_noise():
    """회귀 테스트: 값 차이가 1e-7~1e-8 수준(수치 노이즈)일 때
    min-max 정규화가 이를 0~1로 뻥튀기하면 안 되고, 중립값(0.5)을
    반환해야 한다."""
    noisy_values = [1.376e-07, 1.236e-08, 7.816e-08, 1.083e-07, 6.893e-08]
    normalized = minmax_normalize(noisy_values, noise_floor=1e-3)
    assert all(v == 0.5 for v in normalized), "노이즈 수준의 편차는 중립값으로 처리되어야 한다"


def test_minmax_normalize_preserves_real_signal():
    real_values = [1.0, 5.0, 10.0]
    normalized = minmax_normalize(real_values, noise_floor=1e-3)
    assert normalized[0] == 0.0
    assert normalized[-1] == 1.0
    assert normalized != [0.5, 0.5, 0.5]


def test_repeated_calls_give_consistent_sign():
    """여러 번 반복 호출해도(고유분해가 매번 다시 계산되어도) abs() 처리
    덕분에 부호가 뒤집혀 음수가 나오는 일이 없어야 한다."""
    ds = DocumentStore()
    ds.add_many(["alpha beta gamma", "delta epsilon zeta", "eta theta iota"])
    gs = GraphStore(build_similarity_graph(ds.all()))
    iso = CrossDomainIsomorphismEngine()
    spectral = SpectralRetriever(iso)

    for _ in range(5):
        scores = spectral.score("alpha gamma", ds.all(), gs)
        assert all(s >= 0.0 for s in scores)
