import pytest

from vram_engine.graph.isomorphism import CrossDomainIsomorphismEngine
from vram_engine.graph.verifier import AntiHallucinationVerifier
from vram_engine.interfaces.run_hybrid_demo import LegacyRetrieverAdapter
from vram_engine.safety.circuit_breaker import SelfHealingCircuitBreaker


def _adapter() -> LegacyRetrieverAdapter:
    return LegacyRetrieverAdapter(
        CrossDomainIsomorphismEngine(),
        AntiHallucinationVerifier(),
        SelfHealingCircuitBreaker(),
    )


def test_legacy_adapter_reindex_replaces_the_existing_corpus():
    adapter = _adapter()
    adapter.index(["quantum mechanics wave function"])
    adapter.index(["kimchi stew recipe"])

    assert adapter.documents == ["kimchi stew recipe"]
    assert adapter.doc_adj.shape == (1, 1)

    result = adapter.query("kimchi recipe")
    assert [item["document"] for item in result["results"]] == ["kimchi stew recipe"]


def test_legacy_adapter_requires_index_before_querying():
    with pytest.raises(RuntimeError, match="index"):
        _adapter().query("kimchi recipe")
