import pytest
import torch

from vram_engine.core.document_store import DocumentStore
from vram_engine.core.graph_store import GraphStore
from vram_engine.graph.builder import build_similarity_graph
from vram_engine.graph.isomorphism import CrossDomainIsomorphismEngine
from vram_engine.retrieval.hybrid import minmax_normalize
from vram_engine.retrieval.spectral import SpectralRetriever


def test_spectral_score_is_bounded_and_nonnegative():
    ds = DocumentStore()
    ds.add_many(["quantum wave function collapse", "softmax neural network probability", "kimchi stew recipe korean food"])
    gs = GraphStore(build_similarity_graph(ds.all()))
    spectral = SpectralRetriever(CrossDomainIsomorphismEngine())
    scores = spectral.score("quantum probability collapse", ds.all(), gs)
    assert all(-1e-6 <= s <= 1.0 + 1e-6 for s in scores)


def test_noise_floor_prevents_amplifying_numerical_noise():
    noisy_values = [1.376e-07, 1.236e-08, 7.816e-08, 1.083e-07, 6.893e-08]
    normalized = minmax_normalize(noisy_values, noise_floor=1e-3)
    assert all(v == 0.5 for v in normalized)


def test_minmax_normalize_preserves_real_signal():
    normalized = minmax_normalize([1.0, 5.0, 10.0], noise_floor=1e-3)
    assert normalized[0] == 0.0
    assert normalized[-1] == 1.0
    assert normalized != [0.5, 0.5, 0.5]


def test_repeated_calls_give_consistent_sign():
    ds = DocumentStore()
    ds.add_many(["alpha beta gamma", "delta epsilon zeta", "eta theta iota"])
    gs = GraphStore(build_similarity_graph(ds.all()))
    spectral = SpectralRetriever(CrossDomainIsomorphismEngine())
    for _ in range(5):
        scores = spectral.score("alpha gamma", ds.all(), gs)
        assert all(s >= 0.0 for s in scores)


def test_empty_corpus_returns_no_scores():
    ds = DocumentStore()
    gs = GraphStore(build_similarity_graph(ds.all()))
    spectral = SpectralRetriever(CrossDomainIsomorphismEngine())
    assert spectral.score("anything", ds.all(), gs) == []


def test_top_k_dims_must_be_positive():
    with pytest.raises(ValueError, match="top_k_dims"):
        SpectralRetriever(CrossDomainIsomorphismEngine(), top_k_dims=0)
    with pytest.raises(ValueError, match="top_k_dims"):
        SpectralRetriever(CrossDomainIsomorphismEngine(), top_k_dims=-1)
