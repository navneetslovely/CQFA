"""Finite groups, unitary representations and the group oracle.

Conventions used throughout:

  * a representation is a list {U_g} of unitary matrices closed under
    multiplication, with the identity first;
  * the coherent group projector is Pi = (1/|G|) sum_g U_g, which projects
    onto the G-invariant subspace;
  * the frame register holds n_frame = ceil(log2 |G|) qubits prepared in the
    uniform superposition |G> = |G|^{-1/2} sum_g |g>, carrying amplitude on
    the first |G| computational-basis states and zero on any padding states;
  * the group oracle is V = sum_g |g><g|_frame (x) U_g, a block-diagonal
    unitary on frame (x) data.

Pure NumPy and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# ---------------------------------------------------------------------------
# single-qubit building blocks
# ---------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron(mats):
    """Kronecker product of a list of matrices (left-to-right)."""
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


# ---------------------------------------------------------------------------
# representation container
# ---------------------------------------------------------------------------
@dataclass
class Representation:
    """A finite-group unitary representation {U_g} (identity first)."""

    name: str
    matrices: list          # list[np.ndarray], matrices[0] == identity
    dim: int

    # -- basic descriptors --------------------------------------------------
    @property
    def order(self) -> int:
        return len(self.matrices)

    @property
    def n_frame_qubits(self) -> int:
        return max(1, int(np.ceil(np.log2(self.order))))

    # -- CQFA primitives ----------------------------------------------------
    def projector(self) -> np.ndarray:
        """Coherent group projector Pi = (1/|G|) sum_g U_g."""
        return sum(self.matrices) / self.order

    def frame_superposition(self) -> np.ndarray:
        """|G> = |G|^{-1/2} sum_g |g>  on 2**n_frame_qubits basis states."""
        dimF = 2 ** self.n_frame_qubits
        v = np.zeros(dimF, dtype=complex)
        v[: self.order] = 1.0 / np.sqrt(self.order)
        return v

    def controlled_oracle(self) -> np.ndarray:
        """V = sum_g |g><g| (x) U_g  (block diagonal on frame (x) data)."""
        dimF = 2 ** self.n_frame_qubits
        d = self.dim
        V = np.zeros((dimF * d, dimF * d), dtype=complex)
        for idx in range(dimF):
            block = self.matrices[idx] if idx < self.order else np.eye(d, dtype=complex)
            V[idx * d:(idx + 1) * d, idx * d:(idx + 1) * d] = block
        return V

    # -- sanity checks (used by tests) --------------------------------------
    def is_unitary(self, tol: float = 1e-10) -> bool:
        I = np.eye(self.dim)
        return all(np.allclose(U.conj().T @ U, I, atol=tol) for U in self.matrices)

    def is_closed(self, tol: float = 1e-9) -> bool:
        """Closed under multiplication: every U_g U_h equals some U_k."""
        for A in self.matrices:
            for B in self.matrices:
                P = A @ B
                if not any(np.allclose(P, C, atol=tol) for C in self.matrices):
                    return False
        return True

    def has_identity_first(self, tol: float = 1e-12) -> bool:
        return np.allclose(self.matrices[0], np.eye(self.dim), atol=tol)


# ---------------------------------------------------------------------------
# group construction helpers
# ---------------------------------------------------------------------------
def _perm_matrix(perm) -> np.ndarray:
    """Permutation matrix P with P|i> = |perm[i]>  (column i has a 1 in row perm[i])."""
    n = len(perm)
    P = np.zeros((n, n), dtype=complex)
    for i, pi in enumerate(perm):
        P[pi, i] = 1.0
    return P


def close_matrices(generators, dim, cap: int = 5040, tol: float = 1e-9) -> list:
    """Close a set of generator unitaries into the full finite group.

    Returns the group elements as matrices with the identity first. Convention
    free (works by multiplying matrices), so it is robust for any finite rep.
    """
    I = np.eye(dim, dtype=complex)
    mats = [I]

    def known(P):
        return any(np.allclose(P, C, atol=tol) for C in mats)

    for g in generators:
        if not known(g):
            mats.append(g)

    changed = True
    while changed and len(mats) <= cap:
        changed = False
        for A in list(mats):
            for B in generators:
                P = A @ B
                if not known(P):
                    mats.append(P)
                    changed = True
    return mats


# ---------------------------------------------------------------------------
# concrete group representations
# ---------------------------------------------------------------------------
def z2_rc() -> Representation:
    """G = Z2 (reverse-complement flavour), the RC-EQC case.

    Involution A = X (x) Z on 2 qubits (A@A = I), matching rep_Z2 in
    cqfa_core_verify.py. This is the group whose CQFA(Z2) instance must
    reproduce the reverse-complement equivariance behaviour (regression test).
    """
    A = kron([X, Z])
    return Representation("Z2-rc", close_matrices([A], 4), 4)


def z2_pauli(n_qubits: int = 3) -> Representation:
    """G = Z2 generated by the n-qubit involution X^{(x)n} (balanced +/-1 spectrum).

    n_qubits=3 reproduces rep_Z2_8 of cqfa_advantage.py (Z2 on C^8).
    """
    A = kron([X] * n_qubits)
    return Representation(f"Z2-pauli{n_qubits}", close_matrices([A], 2 ** n_qubits), 2 ** n_qubits)


def cyclic(n: int = 4) -> Representation:
    """Cyclic group C_n = <P>, P the n-cycle shift on C^n (P|i> = |i+1 mod n>)."""
    shift = _perm_matrix([(i + 1) % n for i in range(n)])
    return Representation(f"C{n}", close_matrices([shift], n), n)


def symmetric(k: int = 3) -> Representation:
    """Symmetric group S_k (non-abelian for k>=3), natural permutation rep on C^k.

    Generated by the transposition (0 1) and the k-cycle (0 1 ... k-1).
    """
    transposition = _perm_matrix([1, 0] + list(range(2, k))) if k >= 2 else np.eye(k, dtype=complex)
    kcycle = _perm_matrix([(i + 1) % k for i in range(k)])
    return Representation(f"S{k}", close_matrices([transposition, kcycle], k), k)


def dihedral4() -> Representation:
    """Dihedral group D4 (order 8), symmetries of a square on 4 vertices.

    Generated by rotation r = (0 1 2 3) and reflection s = (1 3) (swap vertices
    1 and 3, fix 0 and 2). Permutation rep on C^4.
    """
    r = _perm_matrix([(i + 1) % 4 for i in range(4)])
    s = _perm_matrix([0, 3, 2, 1])
    return Representation("D4", close_matrices([r, s], 4), 4)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def get_representation(name: str, **kwargs) -> Representation:
    """Dispatch a representation by name.

    Names: 'z2' / 'z2_rc', 'z2_pauli' (kwarg n_qubits), 'cn' (kwarg n) / 'c4',
    'sk' (kwarg k) / 's3', 'd4'.
    """
    key = name.lower()
    if key in ("z2", "z2_rc", "rc"):
        return z2_rc()
    if key in ("z2_pauli", "z2pauli"):
        return z2_pauli(kwargs.get("n_qubits", 3))
    if key in ("cn", "cyclic"):
        return cyclic(kwargs.get("n", 4))
    if key == "c4":
        return cyclic(4)
    if key in ("sk", "symmetric"):
        return symmetric(kwargs.get("k", 3))
    if key == "s3":
        return symmetric(3)
    if key in ("d4", "dihedral4"):
        return dihedral4()
    raise ValueError(f"unknown representation '{name}'")
