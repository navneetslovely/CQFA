
from __future__ import annotations

import numpy as np

from . import dynamical_lie_algebra as D
from . import frame as L


# provenance tokens for the H1 escape flag
QUANTUM = "quantum_nonreprepable"
CLASSICAL = "classical_reprepable"


# ---------------------------------------------------------------------------
# trainability axis  T
# ---------------------------------------------------------------------------
def trainability_axis(rep, n, chi, M, tol=1e-9, cap=None):
 
    d_eff, basis, Bplus = D.frame_induced_dla(rep, n, tol=tol, cap=cap)
    dim_hplus = int(Bplus.shape[1])
    out = dict(d_eff=int(d_eff), dim_hplus=dim_hplus,
               var_lb=0.0, P_g_rho=0.0, P_g_M=0.0)
    if d_eff == 0 or dim_hplus <= 1:
        return out

    chi_r = Bplus.conj().T @ np.asarray(chi, dtype=complex)   # restrict chi to H_+
    nrm = np.linalg.norm(chi_r)
    if nrm < tol:
        # chi has no support in H_+: nothing to train on.
        return out
    chi_r = chi_r / nrm
    rho_r = np.outer(chi_r, chi_r.conj())
    M_r = D.restrict(np.asarray(M, dtype=complex), Bplus)

    P_g_rho = D.g_purity(rho_r, basis)
    P_g_M = D.g_purity(M_r, basis)
    out.update(P_g_rho=P_g_rho, P_g_M=P_g_M,
               var_lb=P_g_rho * P_g_M / float(d_eff))
    return out


# ---------------------------------------------------------------------------
# hardness axis  H
# ---------------------------------------------------------------------------
def hardness_axis(rep, psi, M, source):

    fA, p = L.cqfa_readout(psi, rep, M)
    fB = L.classical_fa_readout(psi, rep, M)
    surplus = 0.0 if fA is None else abs(fA - fB)
    S = L.frame_entropy(psi, rep)
    escape = 1 if source == QUANTUM else 0
    return dict(escape=escape, surplus=float(surplus),
                frame_entropy=float(S), bond_chi=float(2.0 ** S),
                p_norm=float(p))


# ---------------------------------------------------------------------------
# the full certificate for one (model, data) sample
# ---------------------------------------------------------------------------
def certificate(model_name, rep, n, psi, M, source, tol=1e-9, cap=None):

    phi, p = L.coherent_projection(psi, rep)
    chi = phi / np.sqrt(p) if p > 1e-12 else phi
    T = trainability_axis(rep, n, chi, M, tol=tol, cap=cap)
    H = hardness_axis(rep, psi, M, source)
    row = {"model": model_name, "rep": rep.name, "n": int(n),
           "n_frame": int(rep.n_frame_qubits), "source": source,
           "p_norm": H["p_norm"]}
    row.update({"T_" + k: v for k, v in T.items()})
    row.update({"H_" + k: v for k, v in H.items() if k != "p_norm"})
    return row


# ---------------------------------------------------------------------------
# observables for the trainability component: require P_g(M) > 0
# ---------------------------------------------------------------------------
def reachable_observable(rep, n, tol=1e-9):
   
    from .simulator import local_z
    d_eff, basis, Bplus = D.frame_induced_dla(rep, n, tol=tol)
    dim_hplus = Bplus.shape[1]
    if d_eff == 0 or dim_hplus <= 1:
        return local_z(n, 0), False
    # basis[0] is an orthonormal HS vector of an anti-Hermitian element i*H.
    Aanti = basis[0].reshape(dim_hplus, dim_hplus)      # ~ i H on H_+
    Hplus_op = -1j * Aanti                              # Hermitian on H_+
    Hplus_op = (Hplus_op + Hplus_op.conj().T) / 2.0
    # Embed back to full space: M = B (H_+ op) B^dag  (supported on H_+).
    M = Bplus @ Hplus_op @ Bplus.conj().T
    M = (M + M.conj().T) / 2.0
    return M, True
