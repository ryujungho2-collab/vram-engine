"""구 synthesizer.py의 DynamicOperatorRegistry — 그대로 보존.

원래 파일에서도 어디서도 실제 호출되지 않던 죽은 코드였다. Synthesizer의
"검증된 검색결과 → 최종 결과 변환"이라는 새 책임과는 시그니처
(Callable[Tensor, Tensor])가 맞지 않아 억지로 재사용하지 않는다.

TODO(synthesis): 향후 "검증된 결과를 요약/재작성/포맷 변환하는 후처리
오퍼레이터 등록소"로 되살릴 수 있다. 예:
    registry.register("summarize", fn)
    registry.register("rerank", fn)
    op = registry.get("summarize")
    op(final_answer_tensor_repr)
지금 단계에서는 synthesizer.py의 Synthesizer가 이 레지스트리를 사용하지
않는다 — 최소 구현(FinalAnswer 조립)만 제공한다.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Callable


class DynamicOperatorRegistry(nn.Module):
    def __init__(self):
        super().__init__()
        self._registry: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}

    def register(self, op_id: str, operator_fn: Callable[[torch.Tensor], torch.Tensor]):
        self._registry[op_id] = operator_fn

    def get(self, op_id: str) -> Callable[[torch.Tensor], torch.Tensor]:
        return self._registry[op_id]

    def list_operators(self) -> List[str]:
        return list(self._registry.keys())
