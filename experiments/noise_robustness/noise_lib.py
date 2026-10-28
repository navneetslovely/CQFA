
from __future__ import annotations

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


# ---------------------------------------------------------------------------
# single-qubit channels as Kraus operator lists
# ---------------------------------------------------------------------------
def depolarizing(p):
    return [np.sqrt(1 - 3 * p / 4) * I2, np.sqrt(p / 4) * X,
            np.sqrt(p / 4) * Y, np.sqrt(p / 4) * Z]


def bit_flip(p):
    return [np.sqrt(1 - p) * I2, np.sqrt(p) * X]


def phase_flip(p):                                   # dephasing / T2
    return [np.sqrt(1 - p) * I2, np.sqrt(p) * Z]


def bit_phase_flip(p):
    return [np.sqrt(1 - p) * I2, np.sqrt(p) * Y]


def pauli(px, py, pz):
    return [np.sqrt(max(0, 1 - px - py - pz)) * I2,
            np.sqrt(px) * X, np.sqrt(py) * Y, np.sqrt(pz) * Z]


def amplitude_damping(gamma):                        # T1 relaxation
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [K0, K1]


def phase_damping(lam):                              # T2 w/o energy loss
    K0 = np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=complex)
    K1 = np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex)
    return [K0, K1]


def thermal(gamma, nbar):                            # generalized amplitude damping
    p = 1.0 / (1.0 + nbar)
    g = gamma
    K0 = np.sqrt(p) * np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=complex)
    K1 = np.sqrt(p) * np.array([[0, np.sqrt(g)], [0, 0]], dtype=complex)
    K2 = np.sqrt(1 - p) * np.array([[np.sqrt(1 - g), 0], [0, 1]], dtype=complex)
    K3 = np.sqrt(1 - p) * np.array([[0, 0], [np.sqrt(g), 0]], dtype=complex)
    return [K0, K1, K2, K3]


def coherent_z(theta):                               # unitary over-rotation (NOT Kraus mixing)
    U = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)
    return [U]


def coherent_x(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return [np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)]


# ---------------------------------------------------------------------------
# apply a single-qubit channel i.i.d. to qubit q of an n-qubit density matrix
# ---------------------------------------------------------------------------
def _embed(K, q, n):
    ops = [I2] * n
    ops[q] = K
    out = np.array([[1.0 + 0j]])
    for m in ops:
        out = np.kron(out, m)
    return out


def apply_channel_qubit(rho, kraus, q, n):
    return sum(_embed(K, q, n) @ rho @ _embed(K, q, n).conj().T for K in kraus)


def apply_iid(rho, n, kraus):
    """Apply the SAME single-qubit channel independently to every qubit."""
    for q in range(n):
        rho = apply_channel_qubit(rho, kraus, q, n)
    return rho


# ---------------------------------------------------------------------------
# classical readout confusion (measured bitstrings)
# ---------------------------------------------------------------------------
def readout_confusion(q, nbits):
    """N x N confusion matrix T[k',k] = P(measure k' | true k), each bit flips w.p. q."""
    N = 1 << nbits
    T = np.zeros((N, N))
    for k in range(N):
        for k2 in range(N):
            nb = bin(k ^ k2).count("1")
            T[k2, k] = (q ** nb) * ((1 - q) ** (nbits - nb))
    return T


# ---------------------------------------------------------------------------
# named registry for uniform sweeps: model -> (strength -> single-qubit Kraus)
# ---------------------------------------------------------------------------
NOISE_MODELS = {
    "depolarizing":      lambda s: depolarizing(s),
    "dephasing":         lambda s: phase_flip(s),
    "bit_flip":          lambda s: bit_flip(s),
    "bit_phase_flip":    lambda s: bit_phase_flip(s),
    "amplitude_damping": lambda s: amplitude_damping(s),
    "phase_damping":     lambda s: phase_damping(s),
    "thermal(nbar=0.1)": lambda s: thermal(s, 0.1),
    "pauli_asym":        lambda s: pauli(s * 0.5, s * 0.2, s * 0.3),
    "coherent_z":        lambda s: coherent_z(s * np.pi),   # s as fraction of pi over-rotation
    "coherent_x":        lambda s: coherent_x(s * np.pi),
}


def is_trace_preserving(kraus, tol=1e-9):
    """Sanity: sum_K K^dag K == I (CPTP check)."""
    S = sum(K.conj().T @ K for K in kraus)
    return np.allclose(S, np.eye(2), atol=tol)
