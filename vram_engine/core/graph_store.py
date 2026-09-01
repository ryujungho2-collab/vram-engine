"""그래프(인접행렬) 상태 저장소.

인접행렬을 어떻게 만드는지(graph/builder.py)와 언제 롤백하는지
(safety/circuit_breaker.py)는 이 클래스가 모른다. 이 클래스는 오직
"현재 상태를 들고 있다가, 요청받으면 교체하거나 증강된 사본을 만들어준다"만
한다.
"""
from typing import Sequence

import torch


class GraphStore:
    def __init__(self, adjacency: torch.Tensor) -> None:
        self._adjacency = self._validate_adjacency(adjacency)

    def current(self) -> torch.Tensor:
        return self._adjacency

    def replace(self, adjacency: torch.Tensor) -> None:
        adjacency = self._validate_adjacency(adjacency)
        self._adjacency = adjacency.to(
            device=self._adjacency.device,
            dtype=self._adjacency.dtype,
        )

    @property
    def size(self) -> int:
        return self._adjacency.size(0)

    def augmented_with(self, edge_weights: Sequence[float]) -> torch.Tensor:
        """새 노드 1개(마지막 인덱스)를 추가한 (n+1)x(n+1) 그래프를 반환한다.

        원본 인접행렬은 변경하지 않는다(새 텐서를 반환). 질의 노드를
        그래프에 임시로 얹어보는 용도로 쓰인다.
        (기존 hybrid_retriever._augmented_graph와 동일한 로직)
        """
        n = self.size
        if len(edge_weights) != n:
            raise ValueError(f"edge_weights length must match graph size ({n}), got {len(edge_weights)}")
        weights = torch.as_tensor(
            edge_weights,
            device=self._adjacency.device,
            dtype=self._adjacency.dtype,
        )
        if weights.ndim != 1:
            raise ValueError("edge_weights must be one-dimensional")
        if not torch.isfinite(weights).all().item():
            raise ValueError("edge_weights must not contain NaN or Inf")

        aug = torch.eye(n + 1, device=self._adjacency.device, dtype=self._adjacency.dtype)
        aug[:n, :n] = self._adjacency
        aug[n, :n] = weights
        aug[:n, n] = weights
        return aug

    @staticmethod
    def _validate_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
        if not isinstance(adjacency, torch.Tensor):
            raise TypeError("adjacency must be a torch.Tensor")
        if adjacency.ndim != 2:
            raise ValueError("adjacency must be two-dimensional")
        if adjacency.size(0) != adjacency.size(1):
            raise ValueError("adjacency must be a square matrix")
        if not torch.isfinite(adjacency).all().item():
            raise ValueError("adjacency must not contain NaN or Inf")
        return adjacency
