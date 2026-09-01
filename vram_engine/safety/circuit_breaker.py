"""시스템 상태 감시 및 롤백.

NORMAL / WARNING_PURGE / CRITICAL_ROLLBACK 판정 로직과 "가장 최근 안정
상태를 스냅샷으로 유지"하는 정책 모두 원본과 동일 (알고리즘 변경 없음).

경계: 이 클래스는 GraphStore를 알지 못한다. 텐서를 받아 텐서를 반환할 뿐,
GraphStore.replace()를 직접 호출하지 않는다 — 그 결정과 실행은 호출자
(agent/agent.py)의 몫이다.
"""
import torch
from typing import Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SystemSnapshot:
    def __init__(self, state_tensor: torch.Tensor, entropy: float):
        self.state_tensor = state_tensor.clone().detach()
        self.entropy = entropy


class SelfHealingCircuitBreaker:
    def __init__(self, entropy_spike_limit: float = 0.10, max_hallucination_streak: int = 3):
        self.entropy_spike_limit = entropy_spike_limit
        self.max_hallucination_streak = max_hallucination_streak
        self.hallucination_streak = 0
        self.last_stable_snapshot: SystemSnapshot = None

    def update_snapshot(self, state_tensor: torch.Tensor, entropy: float):
        self.last_stable_snapshot = SystemSnapshot(state_tensor, entropy)

    def inspect_and_heal(self, current_entropy: float, prev_entropy: float, is_hallucination: bool, current_state: torch.Tensor) -> Tuple[torch.Tensor, str]:
        delta_h = current_entropy - prev_entropy
        self.hallucination_streak = (self.hallucination_streak + 1) if is_hallucination else 0

        if delta_h > self.entropy_spike_limit or self.hallucination_streak >= self.max_hallucination_streak:
            self.hallucination_streak = 0
            restored_state = self.last_stable_snapshot.state_tensor.clone() if self.last_stable_snapshot else current_state
            return restored_state, "CRITICAL_ROLLBACK"
        elif delta_h > 0.04:
            return current_state, "WARNING_PURGE"
        else:
            # 엔트로피가 최저치를 갱신했는지와 무관하게, NORMAL 상태를 지날
            # 때마다 "가장 최근의 안정 상태"로 스냅샷을 계속 갱신한다.
            self.update_snapshot(current_state, current_entropy)
            return current_state, "NORMAL"
