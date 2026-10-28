
from __future__ import annotations

import numpy as np

from .group import Representation


# ---------------------------------------------------------------------------
# base readout and the two modes
# ---------------------------------------------------------------------------
def base_readout(psi: np.ndarray, M: np.ndarray) -> float:
    """Non-equivariant base model g(psi) = <psi| M |psi>."""
    return float(np.real(psi.conj() @ M @ psi))


def classical_fa_readout(psi: np.ndarray, rep: Representation, M: np.ndarray) -> float:
    """Mode B: classical frame averaging (average the orbit OUTPUTS)."""
    return float(np.mean([base_readout(U @ psi, M) for U in rep.matrices]))


def coherent_projection(psi: np.ndarray, rep: Representation):
    """Return (phi, p) with phi = Pi psi and p = ||phi||^2 (post-selection prob)."""
    phi = rep.projector() @ psi
    p = float(np.real(phi.conj() @ phi))
    return phi, p


def cqfa_readout(psi: np.ndarray, rep: Representation, M: np.ndarray, tol: float = 1e-12):
    """Mode A: coherent amplitude averaging. Returns (value, p).

    value is None exactly when psi is annihilated by Pi (p < tol), as happens for
    fully antisymmetric inputs.
    """
    phi, p = coherent_projection(psi, rep)
    if p < tol:
        return None, p
    return float(np.real(phi.conj() @ M @ phi) / p), p


# ---------------------------------------------------------------------------
# features used by the advantage / learning tasks
# ---------------------------------------------------------------------------
def coherent_feature(psi: np.ndarray, rep: Representation, M: np.ndarray):
    """The CQFA feature c(psi) = <Pi psi|M|Pi psi>/||Pi psi||^2 (None if annihilated)."""
    return cqfa_readout(psi, rep, M)[0]


def diagonal_outputs(psi: np.ndarray, rep: Representation, M: np.ndarray) -> np.ndarray:
    """The per-orbit outputs d_g = <psi| U_g^dag M U_g |psi> that classical FA sees."""
    return np.array([base_readout(U @ psi, M) for U in rep.matrices])


# ---------------------------------------------------------------------------
# order parameter and frame resource
# ---------------------------------------------------------------------------
def symmetric_fraction(psi: np.ndarray, rep: Representation) -> float:
    """p = ||Pi psi||^2 (overlap with the symmetric frame mode)."""
    return coherent_projection(psi, rep)[1]


def frame_asymmetry(psi: np.ndarray, rep: Representation) -> float:
    """q = 1 - p, the monotone control parameter for the coherent surplus."""
    return 1.0 - symmetric_fraction(psi, rep)


def frame_entropy(psi: np.ndarray, rep: Representation) -> float:
    """von Neumann entropy S(rho_R) of the frame register for the pre-measurement
    joint state |Phi> = |G|^{-1/2} sum_g |g> (x) U_g|psi>.
    rho_R[g,h] = (1/|G|) <psi| U_h^dag U_g |psi>.  (The C2 entanglement resource.)"""
    m = rep.order
    mats = rep.matrices
    R = np.array([[psi.conj() @ (mats[h].conj().T @ mats[g]) @ psi
                   for h in range(m)] for g in range(m)]) / m
    R = (R + R.conj().T) / 2.0
    w = np.linalg.eigvalsh(R)
    w = w[w > 1e-12]
    return float(-np.sum(w * np.log2(w)))


# ---------------------------------------------------------------------------
# explicit frame+oracle realisation (ties the abstract oracle to the readout)
# ---------------------------------------------------------------------------
def joint_state(psi: np.ndarray, rep: Representation) -> np.ndarray:
    """|Phi> = V (|G> (x) |psi>) = |G|^{-1/2} sum_g |g> (x) U_g|psi>."""
    joint = np.kron(rep.frame_superposition(), psi)
    return rep.controlled_oracle() @ joint


def postselect_symmetric(psi: np.ndarray, rep: Representation, tol: float = 1e-12):

    phi = joint_state(psi, rep)
    dimF = 2 ** rep.n_frame_qubits
    d = rep.dim
    frame = rep.frame_superposition()
    # <G| (x) I  applied to |Phi>  ->  sum_f conj(frame[f]) * (data block f)
    data = np.zeros(d, dtype=complex)
    for f in range(dimF):
        data += np.conj(frame[f]) * phi[f * d:(f + 1) * d]
    prob = float(np.real(data.conj() @ data))
    if prob < tol:
        return None, prob
    return data, prob
