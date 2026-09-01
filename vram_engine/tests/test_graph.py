import pytest
import torch

from vram_engine.core.document_store import DocumentStore, hybrid_tokenize
from vram_engine.core.graph_store import GraphStore
from vram_engine.graph.builder import build_query_edge_weights, build_similarity_graph


def test_build_similarity_graph_shape_and_symmetry():
    ds = DocumentStore()
    ds.add_many(["a b c", "b c d", "x y z"])
    adj = build_similarity_graph(ds.all(), edge_threshold=0.0)
    assert adj.shape == (3, 3)
    assert torch.allclose(adj, adj.T)


def test_build_similarity_graph_threshold_zeroes_weak_edges():
    ds = DocumentStore()
    ds.add_many(["a b c", "totally different unrelated content"])
    adj = build_similarity_graph(ds.all(), edge_threshold=0.5)
    assert adj[0, 1].item() == 0.0


def test_graph_store_augmented_with_preserves_original():
    base = torch.eye(2)
    gs = GraphStore(base)
    aug = gs.augmented_with([0.3, 0.7])
    assert aug.shape == (3, 3)
    assert torch.equal(gs.current(), base), "augmented_with는 원본을 변경하면 안 된다"
    assert aug[2, 0].item() == pytest.approx(0.3)
    assert aug[2, 1].item() == pytest.approx(0.7)
    assert aug[0, 2].item() == pytest.approx(aug[2, 0].item())


def test_graph_store_replace_preserves_canonical_dtype_and_device():
    gs = GraphStore(torch.eye(2, dtype=torch.float32))
    gs.replace(torch.eye(2, dtype=torch.float64))
    assert gs.current().dtype == torch.float32
    assert gs.current().device.type == "cpu"
    assert gs.augmented_with([0.1, 0.2]).dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_graph_store_normalizes_cuda_rollback_before_next_query():
    gs = GraphStore(torch.eye(2, dtype=torch.float32))
    rollback_state = gs.augmented_with([0.1, 0.2]).to("cuda")
    gs.replace(rollback_state[:2, :2])
    next_query_graph = gs.augmented_with([0.3, 0.4])
    assert gs.current().device.type == "cpu"
    assert next_query_graph.device.type == "cpu"


@pytest.mark.parametrize(
    "adjacency, error",
    [
        ([1.0], TypeError),
        (torch.ones(2, 3), ValueError),
        (torch.tensor([[1.0, float("nan")], [0.0, 1.0]]), ValueError),
    ],
)
def test_graph_store_rejects_invalid_adjacency(adjacency, error):
    with pytest.raises(error):
        GraphStore(adjacency)


def test_graph_store_rejects_invalid_edge_weights():
    gs = GraphStore(torch.eye(2))
    with pytest.raises(ValueError, match="length"):
        gs.augmented_with([0.1])
    with pytest.raises(ValueError, match="NaN"):
        gs.augmented_with([0.1, float("nan")])


def test_build_query_edge_weights_matches_jaccard():
    ds = DocumentStore()
    ds.add_many(["a b", "c d"])
    # 문서 쪽은 hybrid_tokenize(단어+2gram)로 토큰화되므로, 공정한 비교를
    # 위해 질의도 동일한 토크나이저를 거쳐야 자카드 유사도 1.0이 나온다.
    weights = build_query_edge_weights(hybrid_tokenize("a b"), ds.all())
    assert weights[0] == 1.0  # 완전 일치 (동일 토크나이저 사용)
    assert weights[1] == 0.0  # 완전 불일치
