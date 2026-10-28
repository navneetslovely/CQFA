
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from cqfa.group import Representation, close_matrices, kron, X, I2, Z


# ---------------------------------------------------------------------------
# hardware representation registry (Z2, Z2xZ2 only)
# ---------------------------------------------------------------------------
def z2xz2(n: int = 4) -> Representation:

    if n < 2:
        raise ValueError("z2xz2 needs n >= 2 sites")
    Peven = kron([X if s % 2 == 0 else I2 for s in range(n)])
    Podd = kron([X if s % 2 == 1 else I2 for s in range(n)])
    return Representation("z2xz2", close_matrices([Peven, Podd], 2 ** n), 2 ** n)


def hardware_representation(name: str, n_data: int | None = None) -> Representation:
   
    key = name.lower()
    from cqfa import group as G
    if key in ("z2", "z2_rc", "rc"):
        return G.z2_rc()
    if key in ("z2_pauli", "z2pauli"):
        return G.z2_pauli(n_data if n_data is not None else 3)
    if key in ("z2xz2", "z2_z2"):
        return z2xz2(n_data if n_data is not None else 4)
    raise ValueError(
        f"representation '{name}' is not supported on hardware; use one of "
        "z2_rc, z2_pauli, z2xz2 (Z2 / Z2xZ2 only)."
    )


# ---------------------------------------------------------------------------
# qubit-ordering map (numpy site0=MSB  <->  qiskit q0=LSB)
# ---------------------------------------------------------------------------
def site_to_qubit(s: int, n_data: int) -> int:
    """numpy site s (site0 = MSB) -> qiskit data-qubit index (q0 = LSB)."""
    return n_data - 1 - s


def qubit_to_site(q: int, n_data: int) -> int:
    """qiskit data-qubit index q -> numpy site (exact inverse of site_to_qubit)."""
    return n_data - 1 - q


def numpy_to_qiskit_amplitudes(psi: np.ndarray, n_data: int) -> np.ndarray:
    """Reorder a numpy data statevector (site0=MSB) into qiskit order (q0=LSB)."""
    psi = np.asarray(psi, dtype=complex)
    out = np.zeros_like(psi)
    for dd in range(len(psi)):
        bits = [(dd >> (n_data - 1 - s)) & 1 for s in range(n_data)]  # site0..site_{n-1}
        j = 0
        for q in range(n_data):
            j |= bits[n_data - 1 - q] << q                            # qubit q holds site (n-1-q)
        out[j] = psi[dd]
    return out


def qiskit_to_numpy_amplitudes(psi_q: np.ndarray, n_data: int) -> np.ndarray:
    """Reorder a qiskit data statevector (q0=LSB) into numpy order (site0=MSB).

    Exact inverse of :func:`numpy_to_qiskit_amplitudes`.
    """
    psi_q = np.asarray(psi_q, dtype=complex)
    out = np.zeros_like(psi_q)
    for j in range(len(psi_q)):
        qbits = [(j >> q) & 1 for q in range(n_data)]                 # qubit0..qubit_{n-1}
        dd = 0
        for s in range(n_data):
            # site s lives on qubit (n_data-1-s); dd has site0 as MSB
            dd |= qbits[n_data - 1 - s] << (n_data - 1 - s)
        out[dd] = psi_q[j]
    return out


# ---------------------------------------------------------------------------
# register bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class RegisterMap:
    """Which qubits are data / frame / ancilla, plus the site<->qubit map."""

    n_data: int
    n_frame: int
    has_ancilla: bool = False
    data_qubits: list = field(default_factory=list)   # qiskit indices
    frame_qubits: list = field(default_factory=list)
    ancilla_qubit: int | None = None

    def site_qubit(self, s: int) -> int:
        return site_to_qubit(s, self.n_data)


# ---------------------------------------------------------------------------
# per-group controlled oracle V
# ---------------------------------------------------------------------------
def _rep_kind(rep: Representation):
    """Classify a Representation into ('z2_rc' | 'z2_pauli' | 'z2xz2', n_data)."""
    n_data = int(round(np.log2(rep.dim)))
    if 2 ** n_data != rep.dim:
        raise ValueError(f"hardware circuits need power-of-two dim; got dim={rep.dim}")
    name = rep.name.lower()
    if name.startswith("z2xz2"):
        return "z2xz2", n_data
    if name.startswith("z2-rc") or name == "z2_rc":
        return "z2_rc", n_data
    if name.startswith("z2-pauli") or name.startswith("z2_pauli") or name.startswith("z2pauli"):
        return "z2_pauli", n_data
    # Fall back on structure: single non-identity generator that is X^(x)n ?
    raise ValueError(
        f"representation '{rep.name}' has no hardware circuit; supported: "
        "z2_rc, z2_pauli(n), z2xz2(n)."
    )


def _apply_oracle(qc: QuantumCircuit, rep: Representation, data: QuantumRegister,
                  frame: QuantumRegister, n_data: int) -> None:
    """Append the controlled group oracle V for a supported abelian rep.

    Uses only single-control CX/CZ gates (V factorises for abelian Pauli reps).
    """
    kind, _ = _rep_kind(rep)
    if kind == "z2_rc":
        # A = X (x) Z on 2 data qubits: site0 gets X, site1 gets Z.
        qc.cx(frame[0], data[site_to_qubit(0, n_data)])   # controlled-X on site0
        qc.cz(frame[0], data[site_to_qubit(1, n_data)])   # controlled-Z on site1
    elif kind == "z2_pauli":
        # A = X^(x)n : one control, n X-targets.
        for s in range(n_data):
            qc.cx(frame[0], data[site_to_qubit(s, n_data)])
    elif kind == "z2xz2":
        # P_even off frame[0], P_odd off frame[1]; commuting, single-control CX only.
        for s in range(0, n_data, 2):
            qc.cx(frame[0], data[site_to_qubit(s, n_data)])
        for s in range(1, n_data, 2):
            qc.cx(frame[1], data[site_to_qubit(s, n_data)])
    else:  # pragma: no cover
        raise ValueError(f"no oracle for kind {kind}")


# ---------------------------------------------------------------------------
# ansatz W(theta) as native qiskit gates (EXACT match to cqfa.simulator.Ansatz)
# ---------------------------------------------------------------------------
def append_ansatz(qc: QuantumCircuit, data: QuantumRegister, theta, n_data: int,
                  n_layers: int) -> None:

    if theta is None:
        return
    th = np.asarray(theta, dtype=float).reshape(n_layers, n_data, 2)
    for layer in range(n_layers):
        for s in range(n_data):
            q = site_to_qubit(s, n_data)
            qc.ry(float(th[layer, s, 0]), data[q])
            qc.rz(float(th[layer, s, 1]), data[q])
        # CNOT ring in numpy site order: CX(site s -> site s+1 mod n)
        for s in range(n_data):
            qc.cx(data[site_to_qubit(s, n_data)],
                  data[site_to_qubit((s + 1) % n_data, n_data)])


def _infer_n_layers(theta, n_data: int) -> int:
    if theta is None:
        return 0
    length = len(np.asarray(theta).ravel())
    if length % (n_data * 2) != 0:
        raise ValueError(
            f"theta length {length} is not a multiple of n_data*2={n_data * 2}")
    return length // (n_data * 2)


# ---------------------------------------------------------------------------
# Pauli observables measured per-shot as a product of +/-1 eigenvalues
# ---------------------------------------------------------------------------
def pauli_observable(pauli_by_site: dict, n_data: int) -> np.ndarray:
    """Build the numpy matrix (site0=MSB order) for a Pauli string.

    pauli_by_site maps numpy site index -> 'X'|'Y'|'Z'; unlisted sites are I.
    e.g. {0:'X', 1:'Z'} on 2 sites -> X (x) Z. Used as the exact observable M in
    the statevector ground truth so shots and truth agree.
    """
    from cqfa.group import X as PX, Y as PY, Z as PZ, I2 as PI
    table = {"X": PX, "Y": PY, "Z": PZ, "I": PI}
    mats = [table[pauli_by_site.get(s, "I")] for s in range(n_data)]
    return kron(mats)


def _append_basis_rotation(qc, data, pauli_by_site, n_data):
    """Rotate each measured qubit into the Z basis so a computational measurement
    reads the requested Pauli. X -> apply H; Y -> apply Sdg then H; Z -> nothing.
    """
    for s, p in pauli_by_site.items():
        q = site_to_qubit(s, n_data)
        if p == "X":
            qc.h(data[q])
        elif p == "Y":
            qc.sdg(data[q])
            qc.h(data[q])
        elif p == "Z":
            pass
        else:
            raise ValueError(f"unsupported Pauli '{p}'")


# ---------------------------------------------------------------------------
# public builder
# ---------------------------------------------------------------------------
def build_cqfa_circuit(rep: Representation, theta=None, mode: str = "postselect",
                       observable_qubit: int = 0, measure: bool = True,
                       n_layers: int | None = None, observable=None):

    kind, n_data = _rep_kind(rep)
    n_frame = rep.n_frame_qubits
    if n_layers is None:
        n_layers = _infer_n_layers(theta, n_data)

    if mode == "postselect":
        data = QuantumRegister(n_data, "data")
        frame = QuantumRegister(n_frame, "frame")
        rmap = RegisterMap(n_data=n_data, n_frame=n_frame,
                           data_qubits=list(range(n_data)),
                           frame_qubits=list(range(n_data, n_data + n_frame)))
        if measure:
            c_data = ClassicalRegister(n_data, "c_data")
            c_frame = ClassicalRegister(n_frame, "c_frame")
            qc = QuantumCircuit(data, frame, c_data, c_frame)
        else:
            qc = QuantumCircuit(data, frame)

        # |G> on the frame (H on each frame qubit; exact for |G| a power of two)
        for f in range(n_frame):
            qc.h(frame[f])
        # V (controlled oracle)
        _apply_oracle(qc, rep, data, frame, n_data)
        # rotate frame back so post-select |G> <-> frame = 0..0
        for f in range(n_frame):
            qc.h(frame[f])
        # W(theta) on data only
        append_ansatz(qc, data, theta, n_data, n_layers)
        # measurement-basis rotation for a Pauli observable (per-shot +/-1 product)
        if observable:
            _append_basis_rotation(qc, data, observable, n_data)
        if measure:
            qc.measure(data, c_data)
            qc.measure(frame, c_frame)
        return qc, rmap

    if mode == "hadamard":
        # Hadamard test for Re<psi| V_group |psi> with V_group = frame-averaged
        # oracle collapsed to data. We read Re<psi| A |psi> for the involution A
        # (the coherent-interference numerator), NOT f_A itself. Diagnostic only.
        anc = QuantumRegister(1, "anc")
        data = QuantumRegister(n_data, "data")
        rmap = RegisterMap(n_data=n_data, n_frame=0, has_ancilla=True,
                           data_qubits=list(range(1, 1 + n_data)),
                           ancilla_qubit=0)
        if measure:
            c_anc = ClassicalRegister(1, "c_anc")
            qc = QuantumCircuit(anc, data, c_anc)
        else:
            qc = QuantumCircuit(anc, data)
        qc.h(anc[0])
        _apply_controlled_involution(qc, rep, anc[0], data, n_data)
        qc.h(anc[0])
        if measure:
            qc.measure(anc, c_anc)
        return qc, rmap

    raise ValueError(f"unknown mode '{mode}' (use 'postselect' or 'hadamard')")


def _apply_controlled_involution(qc, rep, ctrl, data, n_data):

    kind, _ = _rep_kind(rep)
    if kind == "z2_rc":
        qc.cx(ctrl, data[site_to_qubit(0, n_data)])
        qc.cz(ctrl, data[site_to_qubit(1, n_data)])
    elif kind == "z2_pauli":
        for s in range(n_data):
            qc.cx(ctrl, data[site_to_qubit(s, n_data)])
    elif kind == "z2xz2":
        # product of the two generators = X on every site
        for s in range(n_data):
            qc.cx(ctrl, data[site_to_qubit(s, n_data)])
    else:  # pragma: no cover
        raise ValueError(kind)
