"""기존 main.py 데모 이식. isomorphism/verifier의 알고리즘은 그대로이고,
private 메서드 대신 공개 API(compute_laplacian_spectrum, calc_entropy)를
쓰는 새 모듈 경로만 반영했다.
"""
import torch

from vram_engine.graph.isomorphism import CrossDomainIsomorphismEngine
from vram_engine.graph.verifier import AntiHallucinationVerifier
from vram_engine.synthesis.operator_registry import DynamicOperatorRegistry
from vram_engine.safety.circuit_breaker import SelfHealingCircuitBreaker


def main():
    print("=== VRAM Autonomous Intelligence Engine Starting ===")

    # 1. 초기화
    iso_engine = CrossDomainIsomorphismEngine()
    verifier = AntiHallucinationVerifier()
    registry = DynamicOperatorRegistry()
    circuit_breaker = SelfHealingCircuitBreaker()

    # 2. 샘플 실행 데이터 (양자역학 <-> AI 도메인)
    adj_a = torch.tensor([[1.0, 0.8], [0.8, 1.0]])
    adj_b = torch.tensor([[1.0, 0.85], [0.85, 1.0]])
    matches = iso_engine.match_domains(adj_a, ["Quantum", "Collapse"], adj_b, ["Softmax", "Sampling"])

    print(f"[*] Isomorphic Matches Found: {matches}")

    # 3. 검증 및 서킷 브레이커 테스트
    valid, delta_h, log = verifier.verify_hypothesis(adj_a, 0, 1)
    print(f"[*] Verification Log: {log}")

    print("=== Engine Initialization Complete. Launching GUI Dashboard... ===")
    import tkinter as tk
    from vram_engine.interfaces.dashboard import AutonomousGUIDashboard

    root = tk.Tk()
    app = AutonomousGUIDashboard(root, iso_engine, verifier, circuit_breaker)
    root.mainloop()


if __name__ == "__main__":
    main()
