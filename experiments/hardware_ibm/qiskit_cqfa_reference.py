
from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from cqfa import group as G, frame as L


def build_cqfa_z2_circuit() -> QuantumCircuit:
    """Frame-prep + controlled oracle + coherent-readout rotation (pre-postselection)."""
    qc = QuantumCircuit(3)          # q0,q1 data ; q2 frame
    qc.h(2)                          # |G> on frame
    qc.cx(2, 1)                      # controlled-X on data_hi
    qc.cz(2, 0)                      # controlled-Z on data_lo
    qc.h(2)                          # rotate frame so |+> -> |0> (post-select on q2=0)
    return qc


def qiskit_coherent_readout(psi: np.ndarray, M: np.ndarray):
    """Statevector realisation. Returns (value, p) matching cqfa_layer.cqfa_readout."""
    init = np.zeros(8, dtype=complex)
    for dd in range(4):              # frame=0 block holds psi; index = f*4 + dd, dd=(q1,q0)
        init[dd] = psi[dd]
    sv = Statevector(init).evolve(build_cqfa_z2_circuit())
    amp = sv.data
    data_ps = np.array([amp[idx] for idx in range(8) if ((idx >> 2) & 1) == 0])
    p = float(np.real(data_ps.conj() @ data_ps))
    if p < 1e-12:
        return None, p
    return float(np.real(data_ps.conj() @ M @ data_ps) / p), p


def _selftest(n_trials: int = 50) -> bool:
    rep = G.z2_rc()
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(n_trials):
        psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
        psi /= np.linalg.norm(psi)
        Mm = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        M = (Mm + Mm.conj().T) / 2
        v_np, p_np = L.cqfa_readout(psi, rep, M)
        v_qk, p_qk = qiskit_coherent_readout(psi, M)
        if v_np is None or v_qk is None:
            continue
        worst = max(worst, abs(v_np - v_qk), abs(p_np - p_qk))
    print(f"qiskit vs numpy CQFA(Z2): worst |diff| = {worst:.2e} over {n_trials} trials")
    print("SELFTEST:", "PASS" if worst < 1e-9 else "FAIL")
    return worst < 1e-9


if __name__ == "__main__":
    _selftest()
