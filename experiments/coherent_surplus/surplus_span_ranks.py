"""Certify the coherent-surplus dimension by operator span ranks.

Counting |G| diagonal terms and |G|^2 pair-indexed terms does not by itself
establish that their operator spans have those dimensions: representation
multiplicities, group relations and stabilisers can create dependencies. This
module measures the spans directly.

SCOPE RELATIVE TO surplus_dimension.py
--------------------------------------
results/coherent_surplus/surplus_dimension.csv is produced by
experiments/coherent_surplus/surplus_dimension.py, whose `_rank()` calls
np.linalg.matrix_rank on the stacked vectorised operators, so `coherent_dim`
there is already a numerically computed rank rather than the closed-form count
|G|(|G|-1). Its sample is narrow: one generic Gaussian-Hermitian M at one seed,
and only faithful representations (C_n regular, Z2, S3-natural, D4-natural).
Degenerate M, reducible and multiplicity representations, unfaithful
representations and nontrivial stabilisers are the cases that can break the
closed form, and this module covers them.

WHAT IS COMPUTED
----------------
For a finite group G with unitary rep {U_g}_{g in G} on C^d, and Hermitian M:

    C_{gh} = U_g^dag M U_h
    d_B = dim_R span_R { Herm parts of C_{gg} : g in G }      (classical / diagonal)
    d_A = dim_R span_R { Herm parts of C_{gh} : g,h in G }     (coherent / all pairs)
    coherent surplus = d_A - d_B

Features are <psi|A|psi> with A Hermitian, so the correct object is the REAL
vector space of Hermitian operators (dim_R Herm(d) = d^2). Each Hermitian
operator is mapped by a Hilbert-Schmidt ISOMETRY into R^{d^2}
(diag real parts; sqrt(2)*Re and sqrt(2)*Im of the strict upper triangle),
rows are stacked, and the rank is the numerical rank of that real matrix
via SVD.

Since C_{gh}^dag = U_h^dag M U_g = C_{hg} (M Hermitian), the pair set is closed
under dagger, hence span_C{Herm parts} = span_C{C_{gh}} and the provable bounds are

    d_B <= min(|G|, d^2),      d_A <= min(|G|^2, d^2).

The closed form asserts BOTH bounds are saturated:
    closed_form_paper = min(|G|^2, d^2) - |G|            (as in surplus_dimension.py)
    closed_form_bound = min(|G|^2, d^2) - min(|G|, d^2)  (the bound difference)

TOLERANCE DISCIPLINE (anti tolerance-fiddling)
----------------------------------------------
Every operator is normalised to unit Frobenius norm before stacking (row scaling
does not change a row space) so no operator is spuriously small. Rank is reported
at THREE relative thresholds sigma_i > tau * sigma_max, tau in {1e-6, 1e-9, 1e-12};
`rank_stable` is True only if all three agree. The singular-value spectrum GAP is
reported explicitly: sigma_last_kept, sigma_first_dropped, and their ratio. A rank
is defensible only if the gap is many orders of magnitude.

VACUITY GUARD
-------------
op_norm_scale = max Frobenius norm over the operator set. Any operator with
norm/op_norm_scale < 1e-10 is EXCLUDED as vacuous BEFORE the SVD and counted in
n_vacuous_excluded. If op_norm_scale itself < 1e-14 the whole cell is marked
vacuous_excluded and its rank columns are left EMPTY.

CASES TESTED
------------
(i)   generic Haar-type M            : M_family = gue_generic
(ii)  M with degeneracies            : rank1_proj, half_proj, two_level_degenerate, identity
(iii) REDUCIBLE representations      : rep_type = mult2 (rho (x) I_2, multiplicity 2),
                                       natural perm reps (trivial (+) standard),
                                       unfaithful reps
(iv)  nontrivial stabilisers         : M_family = subgroup_averaged (M is exactly
                                       invariant under a proper nontrivial subgroup H,
                                       so the stabiliser of M under conjugation is >= H);
                                       M_family = identity (stabiliser = all of G)
(v)   abelian AND non-abelian G      : C_3..C_8, Z2 (abelian); S3, S4, D4 (non-abelian)

>= 5 seeds per reported cell (seeds 0..4). Deterministic M families (identity)
are still run at 5 seeds and flagged M_deterministic=True; their std is
legitimately 0. Per-seed rows go to the main CSV; the summary CSV aggregates with
se = std(ddof=1)/sqrt(n).

PRE-REGISTERED CRITERION (two-sided, fixed BEFORE running)
----------------------------------------------------------
CONFIRMS: the numerically computed surplus d_A - d_B equals closed_form_paper
          = min(|G|^2,d^2) - |G| in ALL tested cells. Then the closed form is an
          equality and Proposition (coherent surplus) stands as written.

KILLS:    ANY tested cell has d_A - d_B strictly below closed_form_paper (or d_A
          strictly below min(|G|^2,d^2), or d_B strictly below min(|G|,d^2)).
          Then the counting argument overstates the accessible dimension, the
          closed form is NOT an equality, and the proposition must be restated as
          an UPPER BOUND with the saturation conditions made explicit.

(A cell with surplus ABOVE closed_form_paper is also a failure of the closed form
as an equality; it is recorded separately as `paper_understates` -- it can happen
because closed_form_paper subtracts |G| rather than the true d_B.)

Outputs (both NEW files; surplus_dimension.csv is NOT touched):
    results/coherent_surplus/surplus_span_ranks.csv          (per rep x M_family x seed)
    results/coherent_surplus/surplus_span_ranks_summary.csv  (aggregated over seeds)
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from cqfa import group as G

TOLS = (1e-6, 1e-9, 1e-12)
TOL_REF = 1e-9          # the reported rank uses this relative threshold
VACUOUS_REL = 1e-10     # operator norm ratio below which an operator is vacuous
VACUOUS_ABS = 1e-14     # whole-cell vacuity floor on op_norm_scale
SEEDS = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------------------
# real (Hermitian) vectorisation + numerical rank
# ---------------------------------------------------------------------------
def herm_to_real_vec(H: np.ndarray) -> np.ndarray:
    """Hilbert-Schmidt isometry Herm(d) -> R^{d^2}.

    ||herm_to_real_vec(H)||_2 == ||H||_F exactly.
    """
    d = H.shape[0]
    iu = np.triu_indices(d, 1)
    return np.concatenate([
        np.real(np.diag(H)),
        np.sqrt(2.0) * np.real(H[iu]),
        np.sqrt(2.0) * np.imag(H[iu]),
    ])


def herm_parts(A: np.ndarray):
    """The two Hermitian components of a general operator A."""
    return [(A + A.conj().T) / 2.0, (A - A.conj().T) / (2.0j)]


def span_rank(ops):
    """Numerical dim_R of span_R(ops) for a list of Hermitian ops.

    Returns a dict with ranks at every tolerance in TOLS, the singular-value gap
    around the reference rank, and the vacuity bookkeeping.
    """
    norms = np.array([np.linalg.norm(H) for H in ops])
    scale = float(norms.max()) if len(norms) else 0.0
    out = {"n_ops": len(ops), "op_norm_scale": scale}

    if scale < VACUOUS_ABS:
        out.update(vacuous_excluded=True, n_vacuous_excluded=len(ops),
                   rank=None, sv_last_kept=None, sv_first_dropped=None, sv_gap_ratio=None,
                   rank_stable=None, **{f"rank_tol_{t:g}": None for t in TOLS})
        return out

    keep = [H for H, nrm in zip(ops, norms) if nrm / scale >= VACUOUS_REL]
    out["vacuous_excluded"] = False
    out["n_vacuous_excluded"] = len(ops) - len(keep)

    # unit-normalise each row (does not change the row space, improves conditioning)
    Amat = np.array([herm_to_real_vec(H / np.linalg.norm(H)) for H in keep], dtype=float)
    sv = np.linalg.svd(Amat, compute_uv=False)
    smax = float(sv[0])

    ranks = {}
    for t in TOLS:
        ranks[t] = int((sv > t * smax).sum())
        out[f"rank_tol_{t:g}"] = ranks[t]
    r = ranks[TOL_REF]
    out["rank"] = r
    out["rank_stable"] = (len(set(ranks.values())) == 1)

    # explicit spectrum gap so the rank is defensible, not tolerance-fiddled
    out["sv_last_kept"] = float(sv[r - 1] / smax) if r >= 1 else None
    out["sv_first_dropped"] = float(sv[r] / smax) if r < len(sv) else None
    if r >= 1 and r < len(sv):
        denom = float(sv[r] / smax)
        out["sv_gap_ratio"] = float(sv[r - 1] / sv[r]) if denom > 0 else float("inf")
    else:
        out["sv_gap_ratio"] = float("inf")
    return out


# ---------------------------------------------------------------------------
# representations, including the non-generic ones
# ---------------------------------------------------------------------------
class Case:
    def __init__(self, label, mats, dim, commut, rep_type, note=""):
        self.label, self.mats, self.dim = label, mats, dim
        self.commut, self.rep_type, self.note = commut, rep_type, note
        self.order = len(mats)
        self.irreps = None          # [(d_lambda, m_lambda), ...], filled lazily


def _is_abelian(mats, tol=1e-9):
    for A in mats:
        for B in mats:
            if not np.allclose(A @ B, B @ A, atol=tol):
                return False
    return True


def _mult2(mats):
    """rho (x) I_2 : same irreps, every multiplicity doubled (REDUCIBLE)."""
    return [np.kron(U, np.eye(2, dtype=complex)) for U in mats]


def _blockdiag(blocks):
    n = sum(b.shape[0] for b in blocks)
    out = np.zeros((n, n), dtype=complex)
    i = 0
    for b in blocks:
        k = b.shape[0]
        out[i:i + k, i:i + k] = b
        i += k
    return out


def _s3_irreps(s3_perm_mats):
    """The three irreps of S3, as matrix lists aligned with the natural permutation rep:
    trivial (1x1), sign (1x1), standard (2x2, the permutation action on the sum-zero plane)."""
    triv = [np.eye(1, dtype=complex) for _ in s3_perm_mats]
    sgn = [np.array([[np.round(np.real(np.linalg.det(U)))]], dtype=complex) for U in s3_perm_mats]
    B = np.linalg.qr(np.array([[1, -1, 0], [1, 1, -2]], dtype=float).T)[0]   # 3x2, columns span sum-zero plane
    std = [(B.T @ np.real(U) @ B).astype(complex) for U in s3_perm_mats]
    return triv, sgn, std


def s3_with_multiplicities(s3_perm_mats, m_triv, m_sgn, m_std):
    """m_triv*triv (+) m_sgn*sgn (+) m_std*std of S3 as a block-diagonal representation."""
    triv, sgn, std = _s3_irreps(s3_perm_mats)
    mats = []
    for t, sg, st in zip(triv, sgn, std):
        blocks = [np.kron(np.eye(m), X) for m, X in ((m_triv, t), (m_sgn, sg), (m_std, st)) if m > 0]
        mats.append(_blockdiag(blocks))
    return mats


def isotypic_structure(mats, tol=1e-7, seed=0):
    """(d_lambda, m_lambda) for every irrep occurring in the representation, from the matrices alone.

    The image algebra A = span{U_g} is semisimple, A = (+)_lambda M_{d_lambda} (x) I_{m_lambda}.
    Twirling a random element of A over the group gives a generic element of the centre of A;
    its Hermitian part is constant on each isotypic component, so its eigenprojections P_lambda
    pick out the components. Then rank(P_lambda) = d_lambda m_lambda and dim(A P_lambda) =
    d_lambda^2. Returns None if the two consistency checks (sum d m = d, sum d^2 = dim A) fail.
    """
    n, d = len(mats), mats[0].shape[0]
    rng = np.random.default_rng(seed)
    c = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    a = sum(ci * U for ci, U in zip(c, mats))
    z = sum(U @ a @ U.conj().T for U in mats) / n          # central in A
    H = (z + z.conj().T) / 2                                # centre is *-closed
    w, V = np.linalg.eigh(H)
    scale = max(1.0, float(np.abs(w).max()))
    dimA = int((lambda sv: (sv > 1e-9 * sv[0]).sum())(
        np.linalg.svd(np.array([U.reshape(-1) for U in mats]), compute_uv=False)))
    out, i = [], 0
    while i < d:
        j = i
        while j + 1 < d and abs(w[j + 1] - w[i]) < tol * scale:
            j += 1
        P = V[:, i:j + 1] @ V[:, i:j + 1].conj().T
        r = j + 1 - i                                       # = d_lambda * m_lambda
        AP = np.array([(U @ P).reshape(-1) for U in mats])
        sv = np.linalg.svd(AP, compute_uv=False)
        dsq = int((sv > 1e-9 * sv[0]).sum())                # = d_lambda^2
        dl = int(round(np.sqrt(dsq)))
        if dl * dl != dsq or r % dl:
            return None
        out.append((dl, r // dl))
        i = j + 1
    if sum(dl * ml for dl, ml in out) != d or sum(dl * dl for dl, _ in out) != dimA:
        return None
    return sorted(out)


def pairwise_prediction(irreps):
    """d_A = sum_{lambda,mu} d_l d_m min(d_l d_m, m_l m_m): the coherent count for generic M."""
    return sum(dl * dm * min(dl * dm, ml * mm) for dl, ml in irreps for dm, mm in irreps)


def multiplicity_regime(irreps):
    """'regular' if m = d throughout, 'm_ge_d', 'm_le_d', or 'mixed'."""
    ge = all(m >= dl for dl, m in irreps)
    le = all(m <= dl for dl, m in irreps)
    return "regular" if (ge and le) else "m_ge_d" if ge else "m_le_d" if le else "mixed"


def irrep_string(irreps):
    return ";".join(f"{dl}x{m}" for dl, m in irreps)


def build_cases():
    cases = []

    # ---- (v) abelian, faithful regular reps -------------------------------
    for n in (3, 4, 5, 6, 7, 8):
        r = G.cyclic(n)
        cases.append(Case(f"C{n}", r.matrices, r.dim, "abelian", "regular",
                          "cyclic regular rep (reducible, multiplicity-free)"))
    r = G.z2_rc()
    cases.append(Case("Z2-rc", r.matrices, r.dim, "abelian", "pauli_involution",
                      "X(x)Z involution on C^4"))
    r = G.z2_pauli(3)
    cases.append(Case("Z2-pauli3", r.matrices, r.dim, "abelian", "pauli_involution",
                      "X^(x)3 involution on C^8"))

    # ---- (v) non-abelian, faithful natural permutation reps ---------------
    r = G.symmetric(3)
    s3 = r.matrices
    cases.append(Case("S3-nat", s3, r.dim, "nonabelian", "natural_perm",
                      "S3 on C^3 = trivial (+) standard (REDUCIBLE)"))
    r4 = G.symmetric(4)
    cases.append(Case("S4-nat", r4.matrices, r4.dim, "nonabelian", "natural_perm",
                      "S4 on C^4 = trivial (+) standard (REDUCIBLE)"))
    rd = G.dihedral4()
    cases.append(Case("D4-nat", rd.matrices, rd.dim, "nonabelian", "natural_perm",
                      "D4 on C^4 (REDUCIBLE)"))

    # ---- (iii) REDUCIBLE with MULTIPLICITY 2 : rho (x) I_2 ----------------
    cases.append(Case("C4-mult2", _mult2(G.cyclic(4).matrices), 8, "abelian", "mult2",
                      "C4 regular (x) I_2, every multiplicity doubled"))
    cases.append(Case("S3-nat-mult2", _mult2(s3), 6, "nonabelian", "mult2",
                      "S3 natural (x) I_2: dim of group-algebra image is 5 < |G|=6"))
    cases.append(Case("D4-nat-mult2", _mult2(rd.matrices), 8, "nonabelian", "mult2",
                      "D4 natural (x) I_2"))

    # ---- (iii) MULTIPLICITY PATTERNS of S3: (m_triv, m_sgn, m_std) ---------
    # One-sided patterns (every m >= d_lambda, or every m <= d_lambda) satisfy a closed
    # form, (dim A)^2 or d^2 respectively. Mixed patterns do not, even when {U_g} is
    # linearly independent; (4,1,1) and (3,3,1) are the counterexamples to min(|G|^2,d^2).
    for (mt, ms, md), rtype, note in (
        ((1, 1, 2), "regular_blocks", "S3 regular rep assembled from its irreps: m = d throughout"),
        ((2, 2, 4), "regular_mult2", "two copies of the S3 regular rep: m = 2d throughout"),
        ((1, 1, 1), "multfree_all", "every S3 irrep once: m <= d, so d_A = d^2 = 16"),
        ((1, 0, 3), "mult_uneven", "triv + 3 std, sgn absent: m >= d on occurring irreps, d_A = (dim A)^2 = 25"),
        ((4, 1, 1), "mixed_mult", "4 triv + sgn + std: independent (dim A = 6) and mixed; d_A = 20, not 36"),
        ((3, 3, 1), "mixed_mult", "3 triv + 3 sgn + std: independent (dim A = 6) and mixed; d_A = 24, not 36"),
    ):
        cases.append(Case(f"S3-m{mt}{ms}{md}", s3_with_multiplicities(s3, mt, ms, md),
                          mt + ms + 2 * md, "nonabelian", rtype, note))

    # ---- (iii) UNFAITHFUL reps (group relations kill distinctness) --------
    # C4 acting through its C2 quotient: U_g = X^(g mod 2) on C^2, kernel {0,2}
    Xm = np.array([[0, 1], [1, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    cases.append(Case("C4-thru-C2", [I2, Xm, I2, Xm], 2, "abelian", "unfaithful",
                      "C4 rep with kernel of order 2"))
    # C6 acting through its C3 quotient on C^3
    sh3 = G.cyclic(3).matrices
    cases.append(Case("C6-thru-C3", [sh3[g % 3] for g in range(6)], 3, "abelian", "unfaithful",
                      "C6 rep with kernel of order 2"))
    # trivial rep of C4 on C^4 (maximal degeneracy of the rep itself)
    cases.append(Case("C4-trivial-d4", [np.eye(4, dtype=complex) for _ in range(4)], 4,
                      "abelian", "trivial", "U_g = I for all g in C4, on C^4"))
    # non-abelian unfaithful: S3 through its sign rep on C^2, kernel A3
    sign_mats = []
    for U in s3:
        s = int(np.round(np.real(np.linalg.det(U))))
        sign_mats.append(I2 if s > 0 else Xm)
    cases.append(Case("S3-thru-sign", sign_mats, 2, "nonabelian", "unfaithful",
                      "S3 rep with kernel A3 (order 3)"))
    # faithful but SCALAR image: U_g = i^g I_2 on C^2 (central phases)
    cases.append(Case("C4-scalar-phase", [(1j ** g) * I2 for g in range(4)], 2,
                      "abelian", "scalar_phase",
                      "faithful C4 rep whose image is scalar matrices"))

    return cases


# ---------------------------------------------------------------------------
# M families: (i) generic, (ii) degenerate, (iv) nontrivial stabiliser
# ---------------------------------------------------------------------------
def _proper_subgroup(mats, tol=1e-9):
    """Closure of <U_k> for the smallest k>0 giving a PROPER NONTRIVIAL subgroup.

    Falls back to the full group's first cyclic subgroup if no proper one exists.
    """
    n = len(mats)
    d = mats[0].shape[0]
    best = None
    for k in range(1, n):
        H, U = [np.eye(d, dtype=complex)], mats[k]
        P = U.copy()
        for _ in range(n + 1):
            if np.allclose(P, np.eye(d), atol=tol):
                break
            H.append(P.copy())
            P = P @ U
        if 1 < len(H) < n:
            if best is None or len(H) > len(best):
                best = H
    if best is None:                       # no proper nontrivial cyclic subgroup
        for k in range(1, n):
            H, U = [np.eye(d, dtype=complex)], mats[k]
            P = U.copy()
            for _ in range(n + 1):
                if np.allclose(P, np.eye(d), atol=tol):
                    break
                H.append(P.copy())
                P = P @ U
            if len(H) > 1:
                return H
        return [np.eye(d, dtype=complex)]
    return best


def make_M(family, d, rng, mats):
    """Return (M, is_deterministic, stab_subgroup_order)."""
    if family == "gue_generic":
        A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        return (A + A.conj().T) / 2.0, False, 1
    if family == "rank1_proj":
        v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        v = v / np.linalg.norm(v)
        return np.outer(v, v.conj()), False, 1
    if family == "half_proj":
        k = max(1, d // 2)
        A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        Q, _ = np.linalg.qr(A)
        V = Q[:, :k]
        return V @ V.conj().T, False, 1
    if family == "two_level_degenerate":
        k = max(1, d // 2)
        A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        Q, _ = np.linalg.qr(A)
        V = Q[:, :k]
        return 2.0 * (V @ V.conj().T) - np.eye(d), False, 1
    if family == "identity":
        return np.eye(d, dtype=complex), True, len(mats)
    if family == "subgroup_averaged":
        H = _proper_subgroup(mats)
        A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        M0 = (A + A.conj().T) / 2.0
        M = sum(Uh.conj().T @ M0 @ Uh for Uh in H) / len(H)
        return M, False, len(H)
    raise ValueError(family)


M_FAMILIES = ("gue_generic", "rank1_proj", "half_proj", "two_level_degenerate",
              "identity", "subgroup_averaged")

# which tested case each observable family covers
M_ROLE = {
    "gue_generic": "i_generic_haar",
    "rank1_proj": "ii_degenerate_M",
    "half_proj": "ii_degenerate_M",
    "two_level_degenerate": "ii_degenerate_M",
    "identity": "ii_degenerate_M+iv_max_stabiliser",
    "subgroup_averaged": "iv_nontrivial_stabiliser",
}


# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------
def run_cell(case: Case, family: str, seed: int):
    d, m, Us = case.dim, case.order, case.mats
    rng = np.random.default_rng(1000 * seed + 17)
    M, deterministic, stab_order = make_M(family, d, rng, Us)

    diag_ops = []
    for g in range(m):
        diag_ops += herm_parts(Us[g].conj().T @ M @ Us[g])
    all_ops = []
    for g in range(m):
        Ug = Us[g].conj().T
        for h in range(m):
            all_ops += herm_parts(Ug @ M @ Us[h])

    rb = span_rank(diag_ops)
    ra = span_rank(all_ops)

    bound_dB = min(m, d * d)
    bound_dA = min(m * m, d * d)
    cf_paper = bound_dA - m                       # exactly surplus_dimension.py's formula
    cf_bound = bound_dA - bound_dB

    if case.irreps is None:
        case.irreps = isotypic_structure(Us)
    irr = case.irreps
    pred_pair = pairwise_prediction(irr) if irr else None
    regime = multiplicity_regime(irr) if irr else None

    row = dict(
        group=case.label, order_G=m, dim_d=d, commutativity=case.commut,
        rep_type=case.rep_type, rep_note=case.note,
        M_family=family, M_role=M_ROLE[family], M_deterministic=deterministic,
        stab_subgroup_order=stab_order, seed=seed,
        n_pair_ops=len(all_ops), n_diag_ops=len(diag_ops),
        vacuous_excluded=(rb["vacuous_excluded"] or ra["vacuous_excluded"]),
        n_vacuous_excluded=rb["n_vacuous_excluded"] + ra["n_vacuous_excluded"],
        op_norm_scale_diag=rb["op_norm_scale"], op_norm_scale_pair=ra["op_norm_scale"],
        d_B=rb["rank"], d_A=ra["rank"],
        bound_dB=bound_dB, bound_dA=bound_dA,
        closed_form_paper=cf_paper, closed_form_bound=cf_bound,
        dim_group_algebra=(sum(dl * dl for dl, _ in irr) if irr else None),
        irrep_structure=(irrep_string(irr) if irr else None),
        multiplicity_regime=regime, d_A_pred_pairwise=pred_pair,
        tol_ref=TOL_REF,
        dB_rank_tol_1e6=rb["rank_tol_1e-06"], dB_rank_tol_1e9=rb["rank_tol_1e-09"],
        dB_rank_tol_1e12=rb["rank_tol_1e-12"], dB_rank_stable=rb["rank_stable"],
        dA_rank_tol_1e6=ra["rank_tol_1e-06"], dA_rank_tol_1e9=ra["rank_tol_1e-09"],
        dA_rank_tol_1e12=ra["rank_tol_1e-12"], dA_rank_stable=ra["rank_stable"],
        dB_sv_last_kept=rb["sv_last_kept"], dB_sv_first_dropped=rb["sv_first_dropped"],
        dB_sv_gap_ratio=rb["sv_gap_ratio"],
        dA_sv_last_kept=ra["sv_last_kept"], dA_sv_first_dropped=ra["sv_first_dropped"],
        dA_sv_gap_ratio=ra["sv_gap_ratio"],
    )

    if row["vacuous_excluded"]:
        # rule 7: leave measurements EMPTY, do not let a bound render as a measurement
        row.update(surplus=None, matches_paper=None, matches_bound=None,
                   dB_saturates=None, dA_saturates=None,
                   paper_understates=None, paper_overstates=None,
                   matches_pairwise=None)
        return row

    surplus = ra["rank"] - rb["rank"]
    row.update(
        surplus=surplus,
        matches_paper=(surplus == cf_paper),
        matches_bound=(surplus == cf_bound),
        dB_saturates=(rb["rank"] == bound_dB),
        dA_saturates=(ra["rank"] == bound_dA),
        paper_overstates=(surplus < cf_paper),
        paper_understates=(surplus > cf_paper),
        matches_pairwise=(None if pred_pair is None else ra["rank"] == pred_pair),
    )
    return row


# ---------------------------------------------------------------------------
def run(out_dir):
    cases = build_cases()
    rows = []
    print(f"{'group':>16}{'|G|':>5}{'d':>4}{'M_family':>22}{'d_B':>5}{'bnd':>5}"
          f"{'d_A':>5}{'bnd':>5}{'surp':>6}{'paper':>7}{'ok':>4}{'gapA':>10}")
    for case in cases:
        for fam in M_FAMILIES:
            for seed in SEEDS:
                r = run_cell(case, fam, seed)
                rows.append(r)
            r0 = rows[-1]
            gap = r0["dA_sv_gap_ratio"]
            gs = "inf" if gap == float("inf") else f"{gap:.1e}"
            print(f"{case.label:>16}{case.order:>5}{case.dim:>4}{fam:>22}"
                  f"{str(r0['d_B']):>5}{r0['bound_dB']:>5}{str(r0['d_A']):>5}{r0['bound_dA']:>5}"
                  f"{str(r0['surplus']):>6}{r0['closed_form_paper']:>7}"
                  f"{('Y' if r0['matches_paper'] else 'N'):>4}{gs:>10}")

    os.makedirs(out_dir, exist_ok=True)
    p_main = os.path.join(out_dir, "surplus_span_ranks.csv")
    assert not os.path.exists(p_main), f"refusing to overwrite {p_main}"
    with open(p_main, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", p_main, f"({len(rows)} rows)")

    # ---- aggregate over seeds -------------------------------------------
    keys, agg = [], {}
    for r in rows:
        k = (r["group"], r["M_family"])
        if k not in agg:
            agg[k] = []
            keys.append(k)
        agg[k].append(r)

    srows = []
    for k in keys:
        grp = agg[k]
        r0 = grp[0]
        n = len(grp)
        ok = [r for r in grp if not r["vacuous_excluded"]]
        base = dict(
            group=r0["group"], order_G=r0["order_G"], dim_d=r0["dim_d"],
            commutativity=r0["commutativity"], rep_type=r0["rep_type"],
            M_family=r0["M_family"], M_role=r0["M_role"],
            M_deterministic=r0["M_deterministic"],
            stab_subgroup_order=r0["stab_subgroup_order"],
            n_seeds=n, n_seeds_vacuous=n - len(ok),
            bound_dB=r0["bound_dB"], bound_dA=r0["bound_dA"],
            closed_form_paper=r0["closed_form_paper"],
            closed_form_bound=r0["closed_form_bound"],
            dim_group_algebra=r0["dim_group_algebra"],
            irrep_structure=r0["irrep_structure"],
            multiplicity_regime=r0["multiplicity_regime"],
            d_A_pred_pairwise=r0["d_A_pred_pairwise"],
        )
        if not ok:
            base.update({c: None for c in
                         ("d_B_mean", "d_B_std", "d_B_se", "d_A_mean", "d_A_std", "d_A_se",
                          "surplus_mean", "surplus_std", "surplus_se", "surplus_min",
                          "surplus_max", "all_seeds_match_paper", "any_seed_match_paper",
                          "all_seeds_match_bound", "all_seeds_match_pairwise", "all_ranks_stable",
                          "min_dA_sv_gap_ratio", "min_dB_sv_gap_ratio",
                          "dB_saturates_all", "dA_saturates_all", "verdict")})
            base["verdict"] = "vacuous_excluded"
            srows.append(base)
            continue

        def ms(field):
            v = np.array([r[field] for r in ok], dtype=float)
            sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            return float(v.mean()), sd, sd / np.sqrt(len(v))

        dbm, dbs, dbe = ms("d_B")
        dam, das, dae = ms("d_A")
        spm, sps, spe = ms("surplus")
        sur = [r["surplus"] for r in ok]
        finite_gapA = [r["dA_sv_gap_ratio"] for r in ok if r["dA_sv_gap_ratio"] != float("inf")]
        finite_gapB = [r["dB_sv_gap_ratio"] for r in ok if r["dB_sv_gap_ratio"] != float("inf")]
        all_paper = all(r["matches_paper"] for r in ok)
        base.update(
            d_B_mean=dbm, d_B_std=dbs, d_B_se=dbe,
            d_A_mean=dam, d_A_std=das, d_A_se=dae,
            surplus_mean=spm, surplus_std=sps, surplus_se=spe,
            surplus_min=min(sur), surplus_max=max(sur),
            all_seeds_match_paper=all_paper,
            any_seed_match_paper=any(r["matches_paper"] for r in ok),
            all_seeds_match_bound=all(r["matches_bound"] for r in ok),
            all_seeds_match_pairwise=all(r["matches_pairwise"] for r in ok),
            all_ranks_stable=all(r["dA_rank_stable"] and r["dB_rank_stable"] for r in ok),
            min_dA_sv_gap_ratio=(min(finite_gapA) if finite_gapA else float("inf")),
            min_dB_sv_gap_ratio=(min(finite_gapB) if finite_gapB else float("inf")),
            dB_saturates_all=all(r["dB_saturates"] for r in ok),
            dA_saturates_all=all(r["dA_saturates"] for r in ok),
            verdict=("closed_form_holds" if all_paper else
                     ("closed_form_OVERSTATES" if all(r["paper_overstates"] for r in ok)
                      else "closed_form_MISMATCH")),
        )
        srows.append(base)

    p_sum = os.path.join(out_dir, "surplus_span_ranks_summary.csv")
    assert not os.path.exists(p_sum), f"refusing to overwrite {p_sum}"
    with open(p_sum, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0].keys()))
        w.writeheader()
        w.writerows(srows)
    print("wrote", p_sum, f"({len(srows)} rows)")

    # ---- pre-registered criterion, applied exactly as written ------------
    live = [r for r in rows if not r["vacuous_excluded"]]
    n_over = sum(1 for r in live if r["paper_overstates"])
    n_under = sum(1 for r in live if r["paper_understates"])
    n_dB_fail = sum(1 for r in live if not r["dB_saturates"])
    n_dA_fail = sum(1 for r in live if not r["dA_saturates"])
    n_bound_viol = sum(1 for r in live if r["d_A"] > r["bound_dA"] or r["d_B"] > r["bound_dB"])
    print("\n=== PRE-REGISTERED CRITERION ===")
    print(f"live cells                     : {len(live)} / {len(rows)}")
    print(f"cells where paper OVERSTATES   : {n_over}")
    print(f"cells where paper understates  : {n_under}")
    print(f"cells where d_B < min(|G|,d^2) : {n_dB_fail}")
    print(f"cells where d_A < min(|G|^2,d^2): {n_dA_fail}")
    print(f"BOUND VIOLATIONS (must be 0)   : {n_bound_viol}")
    gen = [r for r in live if r["M_family"] == "gue_generic" and r["matches_pairwise"] is not None]
    print(f"pairwise law d_A = sum d d' min(dd', mm'), generic M : "
          f"{sum(1 for r in gen if r['matches_pairwise'])}/{len(gen)} cells exact")
    mixed = sorted({r["group"] for r in gen if r["multiplicity_regime"] == "mixed"})
    print(f"mixed-multiplicity representations (closed form must fail): {mixed}")
    print(f"all ranks tolerance-stable     : {all(r['dA_rank_stable'] and r['dB_rank_stable'] for r in live)}")
    print("VERDICT:", "CONFIRMS" if (n_over == 0 and n_under == 0) else "KILLS")
    return rows, srows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/coherent_surplus")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    SEEDS = tuple(range(a.seeds))
    run(a.out)
