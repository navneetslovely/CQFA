
from __future__ import annotations

import numpy as np

from .group import I2, X, Y, Z, kron


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def RY(t: float) -> np.ndarray:
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def RZ(t: float) -> np.ndarray:
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]], dtype=complex)


def embed_1q(U: np.ndarray, q: int, n: int) -> np.ndarray:
    return kron([U if i == q else I2 for i in range(n)])


def _apply_1q_state(state: np.ndarray, U: np.ndarray, q: int, n: int) -> np.ndarray:
    """Apply single-qubit gate U to qubit q of an n-qubit statevector (fast)."""
    st = state.reshape([2] * n)
    st = np.tensordot(U, st, axes=([1], [q]))   # new qubit axis at position 0
    st = np.moveaxis(st, 0, q)
    return st.reshape(-1)


def _cnot(control: int, target: int, n: int) -> np.ndarray:
    dim = 2 ** n
    M = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
        if bits[control] == 1:
            bits[target] ^= 1
        j = 0
        for b in bits:
            j = (j << 1) | b
        M[j, i] = 1.0
    return M


def cnot_ring(n: int) -> np.ndarray:
    """Entangler: CNOT(0->1) CNOT(1->2) ... CNOT(n-1->0)."""
    if n == 1:
        return np.eye(2, dtype=complex)
    U = np.eye(2 ** n, dtype=complex)
    for i in range(n):
        U = _cnot(i, (i + 1) % n, n) @ U
    return U


def local_z(n: int, qubit: int = 0) -> np.ndarray:
    """Observable O = Z on `qubit` (eigenvalues +/-1)."""
    return embed_1q(Z, qubit, n)


# ---------------------------------------------------------------------------
# ansatz
# ---------------------------------------------------------------------------
class Ansatz:
   

    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = n_layers * n_qubits * 2
        self._entangler = cnot_ring(n_qubits)

    # -- unitary ------------------------------------------------------------
    def unitary(self, theta: np.ndarray) -> np.ndarray:
        th = np.asarray(theta, dtype=float).reshape(self.n_layers, self.n_qubits, 2)
        W = np.eye(2 ** self.n_qubits, dtype=complex)
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                W = embed_1q(RY(th[layer, q, 0]), q, self.n_qubits) @ W
                W = embed_1q(RZ(th[layer, q, 1]), q, self.n_qubits) @ W
            W = self._entangler @ W
        return W

    # -- apply W(theta) to a state (fast path: gate-by-gate, no full matrix) --
    def apply(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        th = np.asarray(theta, dtype=float).reshape(self.n_layers, self.n_qubits, 2)
        n = self.n_qubits
        st = np.asarray(x, dtype=complex)
        for layer in range(self.n_layers):
            for q in range(n):
                st = _apply_1q_state(st, RY(th[layer, q, 0]), q, n)
                st = _apply_1q_state(st, RZ(th[layer, q, 1]), q, n)
            st = self._entangler @ st
        return st

    # -- expectation <x| W^dag O W |x> -------------------------------------
    def expval(self, theta: np.ndarray, x: np.ndarray, O: np.ndarray) -> float:
        y = self.apply(theta, x)
        return float(np.real(y.conj() @ O @ y))

    # -- analytic parameter-shift gradient (single sample) -----------------
    def expval_grad(self, theta: np.ndarray, x: np.ndarray, O: np.ndarray):
        """Return (value, grad) where grad_i = 1/2[f(theta+pi/2 e_i)-f(theta-pi/2 e_i)]."""
        vals, grads = self.expvals_grad(theta, np.asarray(x).reshape(1, -1), O)
        return float(vals[0]), grads[0]

    # -- batched expectation over a dataset X (rows = states) --------------
    def expvals(self, theta: np.ndarray, X: np.ndarray, O: np.ndarray) -> np.ndarray:
        """<x_n| W^dag O W |x_n> for every row x_n of X. Builds W once."""
        Y = X @ self.unitary(theta).T                    # (N, d) = W applied to each row
        return np.real(np.einsum("ni,ij,nj->n", Y.conj(), O, Y))

    def expvals_grad(self, theta: np.ndarray, X: np.ndarray, O: np.ndarray):
        """Batched value+parameter-shift gradient. Returns (vals (N,), grads (N,P))."""
        theta = np.asarray(theta, dtype=float)
        X = np.atleast_2d(X)
        vals = self.expvals(theta, X, O)
        P = len(theta)
        grads = np.zeros((X.shape[0], P))
        shift = np.pi / 2
        for i in range(P):
            t = theta.copy()
            t[i] += shift
            plus = self.expvals(t, X, O)
            t[i] -= 2 * shift
            minus = self.expvals(t, X, O)
            grads[:, i] = 0.5 * (plus - minus)
        return vals, grads
