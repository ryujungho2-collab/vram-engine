import torch

from vram_engine.safety.circuit_breaker import SelfHealingCircuitBreaker


def test_normal_state_updates_snapshot_every_time():
    """회귀 테스트: '최근 안정 상태' 정책 — NORMAL을 지날 때마다 스냅샷이
    갱신되어야 한다 (엔트로피가 최저치를 갱신했을 때만이 아니라)."""
    cb = SelfHealingCircuitBreaker(entropy_spike_limit=1.0, max_hallucination_streak=99)
    state1 = torch.tensor([[1.0]])
    state2 = torch.tensor([[2.0]])

    _, status1 = cb.inspect_and_heal(current_entropy=0.5, prev_entropy=0.5, is_hallucination=False, current_state=state1)
    assert status1 == "NORMAL"
    assert torch.equal(cb.last_stable_snapshot.state_tensor, state1)

    # 엔트로피가 더 낮아지지 않았지만(오히려 증가), 여전히 NORMAL 범위라면
    # 최신 상태로 스냅샷이 갱신되어야 한다.
    _, status2 = cb.inspect_and_heal(current_entropy=0.52, prev_entropy=0.5, is_hallucination=False, current_state=state2)
    assert status2 == "NORMAL"
    assert torch.equal(cb.last_stable_snapshot.state_tensor, state2), "최신 안정 상태로 갱신되어야 한다"


def test_critical_rollback_restores_last_stable_snapshot():
    """회귀 테스트: CRITICAL_ROLLBACK 발생 시 마지막 안정 스냅샷으로 복원되는지."""
    cb = SelfHealingCircuitBreaker(entropy_spike_limit=0.1, max_hallucination_streak=99)
    stable_state = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    cb.update_snapshot(stable_state, entropy=0.5)

    bad_state = torch.tensor([[9.0, 9.0], [9.0, 9.0]])
    restored, status = cb.inspect_and_heal(
        current_entropy=10.0, prev_entropy=0.5, is_hallucination=False, current_state=bad_state
    )
    assert status == "CRITICAL_ROLLBACK"
    assert torch.equal(restored, stable_state)


def test_hallucination_streak_triggers_rollback():
    cb = SelfHealingCircuitBreaker(entropy_spike_limit=999.0, max_hallucination_streak=2)
    state = torch.tensor([[1.0]])
    cb.update_snapshot(state, entropy=0.1)

    _, status1 = cb.inspect_and_heal(0.1, 0.1, is_hallucination=True, current_state=state)
    assert status1 in ("NORMAL", "WARNING_PURGE")  # 아직 streak=1
    _, status2 = cb.inspect_and_heal(0.1, 0.1, is_hallucination=True, current_state=state)
    assert status2 == "CRITICAL_ROLLBACK"  # streak=2 도달


def test_warning_purge_does_not_roll_back():
    cb = SelfHealingCircuitBreaker(entropy_spike_limit=1.0, max_hallucination_streak=99)
    state = torch.tensor([[1.0]])
    _, status = cb.inspect_and_heal(current_entropy=0.5, prev_entropy=0.4, is_hallucination=False, current_state=state)
    assert status == "WARNING_PURGE"
