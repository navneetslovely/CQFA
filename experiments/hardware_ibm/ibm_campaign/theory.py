"""
experiments.hardware_ibm.ibm_campaign.theory
=========================================
EXACT ground truth for every campaign experiment, computed from the verified
numpy core (``cqfa.frame``, ``experiments.spt_measurement_cost.spt_model``) and exact
statevectors. Every PUB stores its theory value here at build time, so the
offline dry run can assert estimate == theory within shot noise (the
correctness gate) and the IBM run can report abs-error against a fixed truth.

No shots, no sampling — pure linear algebra.
"""
from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Statevector

from cqfa import frame as L
from cqfa.group import Representation
from ..cqfa_circuit import qiskit_to_numpy_amplitudes, pauli_observable
from ..readout import statevector_readout


# ---------------------------------------------------------------------------
# C1 — SPT string order  <O_string> = <prod_i K_i>
# ---------------------------------------------------------------------------
def spt_string_label(n: int) -> dict:
    """Y-decorated correct SPT order parameter Z_0 Y_1 X_2..X_{n-3} Y_{n-2} Z_{n-1}
    as {site: Pauli}, telescoped from prod_{i=1}^{n-2} K_i (see e2_spt)."""
    from experiments.spt_measurement_cost.readout_shadow import string_pauli_label
    return string_pauli_label(n)


def c1_string_theory(prep_circuit, n: int) -> float:
    """Exact expectation of the ACTUAL measured Pauli string for the prepared state.

    The campaign measures the tensor-product Y-decorated string
    T = Z_0 Y_1 X_2..X_{n-3} Y_{n-2} Z_{n-1} (from ``spt_string_label``) by per-site
    basis rotation + parity. The telescoped stabilizer product prod K_i equals T
    only up to a global sign s in {+1,-1} that alternates with n (from the Z.X=-iY
    phases), so we must contract against T itself, not prod K_i: the cluster state
    is a +/-1 eigenstate of T (|<T>|=1), the paramagnet gives <T>=0. This is exactly
    what the +/-1 parity readout returns, so estimate and theory agree by
    construction (verified by the offline gate).
    """
    from experiments.spt_measurement_cost import spt_model as Msp
    mats = {"X": Msp.X, "Y": Msp.Y, "Z": Msp.Z}
    label = spt_string_label(n)
    T = Msp.op_on({s: mats[p] for s, p in label.items()}, n)
    sv = np.asarray(Statevector.from_instruction(prep_circuit).data, dtype=complex)
    # C1 circuits map qubit q -> site q (built fresh, no site_to_qubit remap),
    # but qiskit amplitudes are q0=LSB while op_on uses site0=MSB.
    psi = qiskit_to_numpy_amplitudes(sv, n)
    return float(np.real(psi.conj() @ T @ psi))


# ---------------------------------------------------------------------------
# C2 — two-copy purity  Tr(rho^2)  via destructive Bell/SWAP
# ---------------------------------------------------------------------------
def c2_purity_theory_mixed(theta: float) -> float:
    """Purity of the 1-qubit state rho = diag(cos^2 t, sin^2 t) built by
    Ry(2t) then CNOT into an environment qubit:  Tr(rho^2) = 1 - 0.5 sin^2(2t)."""
    return float(1.0 - 0.5 * np.sin(2.0 * theta) ** 2)


def c2_purity_theory_from_state(prep_circuit, sys_qubits, env_qubits) -> float:
    """Exact Tr(rho_sys^2) for the reduced state of ``sys_qubits`` after tracing
    out ``env_qubits`` (ground truth for an arbitrary C2 prep). Uses the density
    matrix partial trace; only used at build time on tiny (<=4 qubit) circuits."""
    from qiskit.quantum_info import partial_trace, DensityMatrix
    dm = DensityMatrix(Statevector.from_instruction(prep_circuit))
    keep = list(sys_qubits)
    trace_out = [q for q in range(prep_circuit.num_qubits) if q not in keep]
    rho = partial_trace(dm, trace_out) if trace_out else dm
    return float(np.real(np.trace(rho.data @ rho.data)))


# ---------------------------------------------------------------------------
# C3/C5/C6 — CQFA coherent post-selected readout f_A (exact)
# ---------------------------------------------------------------------------
def cqfa_readout_theory(statevector_circuit, n_data: int, n_frame: int,
                        observable: dict):
    """Exact post-selected coherent readout (f_A, p) of the measure=False CQFA
    circuit, using the numpy observable M (matches ``readout_from_counts``)."""
    M = pauli_observable(observable, n_data)
    f, p = statevector_readout(
        Statevector.from_instruction(statevector_circuit), n_data, n_frame, M=M)
    return f, p


# ---------------------------------------------------------------------------
# C4 — coherent surplus  f_A - f_B  (Mode A minus classical frame averaging)
# ---------------------------------------------------------------------------
def surplus_theory(psi: np.ndarray, rep: Representation, observable: dict, n_data: int):
    """Exact (f_A, f_B, surplus) for data state psi (numpy site0=MSB order).

    f_A = <Pi psi|M|Pi psi>/||Pi psi||^2   (Mode A, coherent)
    f_B = (1/|G|) sum_g <U_g psi|M|U_g psi> (Mode B, classical frame averaging)
    """
    M = pauli_observable(observable, n_data)
    f_A, p = L.cqfa_readout(psi, rep, M)
    f_B = L.classical_fa_readout(psi, rep, M)
    surplus = None if f_A is None else float(f_A - f_B)
    return f_A, f_B, surplus, p


def modeB_arm_theory(psi: np.ndarray, U_g: np.ndarray, observable: dict, n_data: int) -> float:
    """Exact <U_g psi| M |U_g psi> for a single Mode-B orbit arm (one group element)."""
    M = pauli_observable(observable, n_data)
    return L.base_readout(U_g @ psi, M)


# ---------------------------------------------------------------------------
# C8 — Mode F (frame Fourier readout) hidden-shift recovery vs blind Mode B
# ---------------------------------------------------------------------------
def c8_frame_distribution_theory(prep_circuit_no_measure, n_frame: int) -> np.ndarray:
    """Exact frame-register outcome distribution P(k) of the *built* C8 circuit
    (prep with NO measurement), by exact statevector.

    The circuit acts on exactly ``n_frame`` qubits (q0 = LSB), so amplitude index
    == integer k = sum_j bit_j 2^j, matching the measured bitstring int(...) decode.
    Computing theory from the built circuit's own statevector (as C1 does) means the
    offline gate catches any QFT/oracle convention slip before a device is touched:
      Mode F -> a delta at k = s (Fourier readout recovers the hidden shift);
      Mode B -> uniform (classical frame averaging is blind to the cross-frame phase).
    """
    sv = np.asarray(Statevector.from_instruction(prep_circuit_no_measure).data, dtype=complex)
    assert sv.shape[0] == 2 ** n_frame, (sv.shape, n_frame)
    return np.abs(sv) ** 2
