
from __future__ import annotations

import numpy as np

from .group import X, Y, Z  # noqa: F401  (X kept for completeness / callers)
from .simulator import cnot_ring, embed_1q


# ---------------------------------------------------------------------------
# Lie-closure engine (Larocca 2022 Algorithm 1, HS-vectorised)
# ---------------------------------------------------------------------------
def _vec(M: np.ndarray) -> np.ndarray:
    return np.asarray(M).reshape(-1).astype(complex)


def _orthonormal_add(basis: list, M: np.ndarray, tol: float = 1e-9) -> bool:
   
    v = _vec(M)
    for b in basis:
        v = v - (b.conj() @ v) * b
    nv = np.linalg.norm(v)
    if nv > tol:
        basis.append(v / nv)
        return True
    return False


def lie_closure_dim(generators, tol: float = 1e-9, cap: int | None = None):
 
    generators = [np.asarray(g, dtype=complex) for g in generators]
    if len(generators) == 0:
        return 0, np.zeros((0, 0), dtype=complex)
    d2 = generators[0].size
    cap = d2 if cap is None else int(cap)

    basis: list = []
    frontier = [g for g in generators if _orthonormal_add(basis, g, tol)]
    while frontier and len(basis) < cap:
        new_frontier = []
        for A in generators:
            for B in frontier:
                if len(basis) >= cap:
                    break
                C = A @ B - B @ A                       # Lie bracket [A, B]
                if np.linalg.norm(C) > tol and _orthonormal_add(basis, C, tol):
                    new_frontier.append(C)
        frontier = new_frontier
    dim = min(len(basis), cap)
    return dim, np.array(basis[:cap]) if basis else np.zeros((0, d2), dtype=complex)


# ---------------------------------------------------------------------------
# generator sets for the hardware-efficient ansatz
# ---------------------------------------------------------------------------
def rotation_generators(n: int):

    gens = []
    for q in range(n):
        gens.append(1j * embed_1q(Y, q, n))
        gens.append(1j * embed_1q(Z, q, n))
    return gens


def hea_circuit_generators(n: int, include_entangler: bool = True):
  
    base = rotation_generators(n)
    if include_entangler and n > 1:
        E = cnot_ring(n)
        base = base + [E @ g @ E.conj().T for g in base]
    return base


# ---------------------------------------------------------------------------
# the frame-induced (restricted, commutant) DLA -- the Representation-Dial knob
# ---------------------------------------------------------------------------
def symmetrized_generators(generators, rep):
  
    Us = rep.matrices
    m = rep.order
    return [sum(U @ g @ U.conj().T for U in Us) / m for g in generators]


def invariant_subspace_basis(rep, tol: float = 1e-9) -> np.ndarray:
  
    Pi = rep.projector()
    w, V = np.linalg.eigh((Pi + Pi.conj().T) / 2.0)
    return V[:, w > 1.0 - tol]


def restrict(op: np.ndarray, Bplus: np.ndarray) -> np.ndarray:
    """Compress an operator to the H_+ block: B^dag op B."""
    return Bplus.conj().T @ op @ Bplus


def _traceless(H: np.ndarray) -> np.ndarray:
    d = H.shape[0]
    return H - (np.trace(H) / d) * np.eye(d, dtype=complex)


def frame_induced_dla(rep, n: int, tol: float = 1e-9, cap: int | None = None):

    Bplus = invariant_subspace_basis(rep, tol)
    dim_hplus = Bplus.shape[1]
    if dim_hplus <= 1:
        # H_+ is 0- or 1-dimensional: su is trivial.
        return 0, np.zeros((0, dim_hplus ** 2), dtype=complex), Bplus
    gens = symmetrized_generators(hea_circuit_generators(n), rep)
    rgens = [_traceless(restrict(g, Bplus)) for g in gens]
    d_eff, basis = lie_closure_dim(rgens, tol=tol, cap=cap)
    return d_eff, basis, Bplus


# ---------------------------------------------------------------------------
# g-purity and the exact Ragone-Fontana variance law
# ---------------------------------------------------------------------------
def g_purity(H: np.ndarray, dla_basis: np.ndarray) -> float:
   
    if dla_basis is None or dla_basis.shape[0] == 0:
        return 0.0
    H = _traceless(np.asarray(H, dtype=complex))
    coeffs = dla_basis.conj() @ H.reshape(-1)
    return float(np.sum(np.abs(coeffs) ** 2))


def ragone_variance(rho: np.ndarray, O: np.ndarray, dla_basis: np.ndarray,
                    d_eff: int) -> float:
 
    if d_eff <= 0 or dla_basis is None or dla_basis.shape[0] == 0:
        return 0.0
    return g_purity(rho, dla_basis) * g_purity(O, dla_basis) / float(d_eff)
