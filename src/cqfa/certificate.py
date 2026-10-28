"""The Representation Dial certificate: a computable (T, H) pair.

For a model and a data provenance the certificate is a point

    C(model, data) = ( T , H )        in  R x R_{>=0}

  * T is the trainability component, T = log10(Var_lb), where the
    gradient-variance law Var_lb = P_g(rho_chi) P_g(M) / dim(g_G) is evaluated
    on the frame-induced restricted algebra g_G of ``dynamical_lie_algebra``.
    The mechanism is proven for the Z2 special case, where dim(g_G) has a
    closed polynomial form; for the wider family it is measured. Higher T
    corresponds to a flatter-plateau-free regime.

  * H is the hardness component, formed from two parts:

      H1, escape: a binary flag equal to 1 when the input states are supplied
          as non-re-preparable quantum copies, so that a classical
          data-acquisition phase is not executable. Unconditional under that
          premise; it invokes no classical-hardness conjecture.

      H2, witness: a continuous simulation-cost witness, the frame-cut
          entanglement S_frame, which sets the bond dimension
          chi = 2^S_frame needed to represent the coherent readout across the
          frame-data cut, together with the coherent surplus |f_A - f_B|.
          This is a resource witness that grows with the oracle, not a
          hardness theorem.

Both components are computed from the same model object, and different design
choices set them separately, so a candidate design can be certified before any
circuit is run.

References
----------
Ragone et al. (2024); Fontana et al. (2024) (variance law).
Larocca et al. (2022) (Lie closure).
Cerezo et al., Nat. Commun. (2025) (classical data-acquisition trap).
Meyer et al., PRX Quantum 4, 010328 (2023) (equivariant commutant ansatz).
"""
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
    """Compute the trainability coordinate on the frame-induced restricted DLA.

    Parameters
    ----------
    rep : Representation
    n   : int, number of data qubits (rep.dim == 2**n)
    chi : (rep.dim,) complex, the coherent-projected input chi = Pi psi / ||.||
    M   : (rep.dim,) x (rep.dim,) Hermitian observable
    """
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
    """Compute the hardness coordinate: escape flag + simulation-cost proxies.

    surplus     = |f_A - f_B|  (coherent interference; 0 for a classical/output
                  model by construction) -- H2a.
    frame_entropy = S_vN(rho_R) across the frame|data cut -- H2b.
    bond_chi    = 2**frame_entropy (effective Schmidt rank / MPS bond dim).
    escape      = 1 iff source is non-re-preparable quantum data -- H1.
    """
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
    """Full certificate point for one input state `psi`.

    For the CQFA / equivariant model the trainability axis is evaluated on the
    frame-induced restricted DLA g_G and the coherent-projected input chi.  For
    the generic (non-equivariant) model the caller should instead read the
    ambient DLA (dim 4^n - 1) directly -- see run_certificate.py, which sets
    the T coordinate per-model.  Here we always report the FRAME-INDUCED T so
    the certificate object is uniform; the ambient/barren case is a separate
    column produced by the driver.
    """
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
    """Build a Hermitian observable M (full 2**n-dim) whose H_+ restriction has
    non-zero g-purity, so the restricted model can actually move the loss.

    Construction: take a generator of the restricted DLA g_G, lift it back to
    H_+ as a Hermitian operator, and embed into the full space via the H_+
    basis.  Guarantees P_{g_G}(M) > 0.  Returns (M, ok) where ok is False
    if H_+ is too small to admit a non-trivial DLA (then M defaults to Z_0).
    """
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
