import torch

from vram_engine.graph.verifier import AntiHallucinationVerifier


def test_calc_entropy_is_nonnegative():
    verifier = AntiHallucinationVerifier()
    adj = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    entropy = verifier.calc_entropy(adj)
    assert entropy.item() >= 0.0


def test_verify_hypothesis_rejects_large_entropy_spike():
    verifier = AntiHallucinationVerifier(entropy_tolerance=0.001, divergence_limit=0.001)
    adj = torch.eye(4)  # 완전히 고립된 노드들 -> 엣지 추가가 큰 변화를 만듦
    is_valid, delta_h, log = verifier.verify_hypothesis(adj, 0, 1)
    assert is_valid is False
    assert "REJECTED" in log


def test_verify_hypothesis_accepts_stable_change():
    verifier = AntiHallucinationVerifier(entropy_tolerance=10.0, divergence_limit=10.0)
    adj = torch.eye(4)
    is_valid, delta_h, log = verifier.verify_hypothesis(adj, 0, 1)
    assert is_valid is True
    assert "VERIFIED" in log


def test_verify_candidates_returns_one_result_per_candidate():
    verifier = AntiHallucinationVerifier()
    adj = torch.eye(4)
    results = verifier.verify_candidates(adj, query_node_idx=0, candidate_doc_ids=[1, 2])
    assert len(results) == 2
    assert {r.doc_id for r in results} == {1, 2}
