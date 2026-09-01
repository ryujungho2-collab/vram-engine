import pytest
import torch

from vram_engine.graph.isomorphism import CrossDomainIsomorphismEngine


def test_isomorphism_rejects_non_positive_tau():
    with pytest.raises(ValueError, match="tau"):
        CrossDomainIsomorphismEngine(tau=0)


def test_isomorphism_rejects_label_count_mismatch():
    engine = CrossDomainIsomorphismEngine()
    adj = torch.eye(2)
    with pytest.raises(ValueError, match="labels"):
        engine.match_domains(adj, ["only-one"], adj, ["a", "b"])
