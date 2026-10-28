"""Test a corrected closed form for the coherent-surplus dimension.

`surplus_span_ranks.py` establishes that the closed form
    surplus = min(|G|^2, d^2) - |G|
is an upper bound rather than an equality: it fails on reducible and
multiplicity representations, on unfaithful and scalar-image representations,
and on degenerate M. This module tests a replacement law suggested by the
structure of those failures.

DERIVATION (why |G| is the wrong quantity)
------------------------------------------
Let A = span_C{U_g : g in G} subset M_d(C) -- the image of the group algebra
C[G] under the representation. Then

    span_C{ U_g^dag M U_h : g,h in G } = A M A       (A is dagger-closed)

so the pair span is governed by dim A, NOT by |G|. By Artin-Wedderburn,
dim A = sum over the DISTINCT irreps pi actually appearing in the rep of
(dim pi)^2. Two consequences the counting argument misses:

  * multiplicity is invisible to A: rho and rho (x) I_2 have the SAME dim A,
    so doubling the multiplicity does not add a single accessible dimension;
  * an unfaithful rep has dim A = dim of the image of C[G/ker], strictly less
    than |G|; a scalar-image rep has dim A = 1 whatever |G| is.

dim A = |G| holds iff EVERY irrep of G appears in the rep (e.g. the regular
rep), which is the regime in which the closed form is exact.

CANDIDATE CORRECTED LAW (tested here):
    d_A_pred = min( (dim A)^2, d^2 )      for generic M
    d_B_pred = min( dim A,     d^2 )      for generic M          [see caveat]
    surplus_pred = d_A_pred - d_B_pred

PAIRWISE LAW (also tested here; the exact count)
------------------------------------------------
With the isotypic decomposition C^d = (+)_lambda C^{d_lambda} (x) C^{m_lambda},
    d_A = sum_{lambda,mu} d_lambda d_mu min(d_lambda d_mu, m_lambda m_mu)      for generic M.
The (dim A)^2 candidate is its special case m_lambda >= d_lambda for every occurring irrep,
and d^2 is the special case m_lambda <= d_lambda. Mixed multiplicities fall strictly below
both, so min((dim A)^2, d^2) overstates d_A even when {U_g} is linearly independent; the
S3 patterns (4,1,1) and (3,3,1) in the sweep are such cases. Irrep dimensions and
multiplicities are computed numerically from the matrices (isotypic_structure), not
entered by hand.

PRE-REGISTERED CRITERION (two-sided, fixed BEFORE running)
----------------------------------------------------------
CONFIRMS: d_A == min((dim A)^2, d^2) in every generic-M cell (M_family =
          gue_generic, 5 seeds). The corrected law is then exact on the tested
          set.
KILLS:    any generic-M cell with d_A != min((dim A)^2, d^2). The corrected law
          is then only an upper bound and is reported as such.

Separately (NOT part of the pass/fail criterion, reported as measurement only):
the same predictions are evaluated on degenerate M, where they are EXPECTED to
fail, since A M A genuinely depends on M -- e.g. M = I gives A M A = A, so
d_A = dim A rather than (dim A)^2. Those rows are labelled generic_M=False and
carry a separate column d_A_equals_dimA (the M = I law).

>= 5 seeds per cell; per-seed rows, plus paired per-seed deltas
(d_A - d_A_pred) differenced FIRST then averaged, se = std(ddof=1)/sqrt(n).
Vacuity guard inherited from span_rank().

Writes results/coherent_surplus/surplus_corrected_law.csv.

Run::

    python -m experiments.coherent_surplus.surplus_corrected_law
"""
from __future__ import annotations

import csv
import os

import numpy as np

from experiments.coherent_surplus.surplus_span_ranks import (
    build_cases, make_M, herm_parts, span_rank, M_FAMILIES, SEEDS,
    isotypic_structure, pairwise_prediction, multiplicity_regime, irrep_string,
)


def dim_group_algebra(Us, tol=1e-9):
    V = np.array([U.reshape(-1) for U in Us])
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    sv = np.linalg.svd(V, compute_uv=False)
    return int((sv > tol * sv[0]).sum())


def main(out_dir="results/coherent_surplus"):
    rows = []
    for case in build_cases():
        d, m, Us = case.dim, case.order, case.mats
        dimA = dim_group_algebra(Us)
        if case.irreps is None:
            case.irreps = isotypic_structure(Us)
        irr = case.irreps
        pred_pair = pairwise_prediction(irr) if irr else None
        for fam in M_FAMILIES:
            for seed in SEEDS:
                rng = np.random.default_rng(1000 * seed + 17)
                M, det, stab = make_M(fam, d, rng, Us)
                dh, ph = [], []
                for g in range(m):
                    dh += herm_parts(Us[g].conj().T @ M @ Us[g])
                for g in range(m):
                    Ug = Us[g].conj().T
                    for h in range(m):
                        ph += herm_parts(Ug @ M @ Us[h])
                rb, ra = span_rank(dh), span_rank(ph)
                dA_pred = min(dimA * dimA, d * d)
                dB_pred = min(dimA, d * d)
                paper_dA = min(m * m, d * d)
                r = dict(
                    group=case.label, order_G=m, dim_d=d,
                    commutativity=case.commut, rep_type=case.rep_type,
                    M_family=fam, generic_M=(fam == "gue_generic"),
                    M_deterministic=det, stab_subgroup_order=stab, seed=seed,
                    dim_group_algebra=dimA, dimA_equals_order_G=(dimA == m),
                    irrep_structure=(irrep_string(irr) if irr else None),
                    multiplicity_regime=(multiplicity_regime(irr) if irr else None),
                    d_A_pred_pairwise=pred_pair,
                    vacuous_excluded=(rb["vacuous_excluded"] or ra["vacuous_excluded"]),
                    d_A=ra["rank"], d_B=rb["rank"],
                    d_A_pred_corrected=dA_pred, d_B_pred_corrected=dB_pred,
                    surplus_pred_corrected=dA_pred - dB_pred,
                    d_A_pred_paper=paper_dA, surplus_pred_paper=paper_dA - m,
                    dA_rank_stable=ra["rank_stable"], dB_rank_stable=rb["rank_stable"],
                    dA_sv_gap_ratio=ra["sv_gap_ratio"], dB_sv_gap_ratio=rb["sv_gap_ratio"],
                )
                if r["vacuous_excluded"]:
                    r.update(delta_dA_corrected=None, delta_dB_corrected=None,
                             delta_surplus_corrected=None, delta_surplus_paper=None,
                             corrected_law_exact=None, paper_law_exact=None,
                             d_A_equals_dimA=None,
                             delta_dA_pairwise=None, pairwise_law_exact=None)
                else:
                    surp = ra["rank"] - rb["rank"]
                    r.update(
                        surplus=surp,
                        delta_dA_corrected=ra["rank"] - dA_pred,
                        delta_dB_corrected=rb["rank"] - dB_pred,
                        delta_surplus_corrected=surp - (dA_pred - dB_pred),
                        delta_surplus_paper=surp - (paper_dA - m),
                        corrected_law_exact=(ra["rank"] == dA_pred),
                        paper_law_exact=(surp == paper_dA - m),
                        d_A_equals_dimA=(ra["rank"] == dimA),
                        delta_dA_pairwise=(None if pred_pair is None else ra["rank"] - pred_pair),
                        pairwise_law_exact=(None if pred_pair is None else ra["rank"] == pred_pair),
                    )
                rows.append(r)

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "surplus_corrected_law.csv")
    assert not os.path.exists(p), f"refusing to overwrite {p}"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", p, f"({len(rows)} rows)")

    live = [r for r in rows if not r["vacuous_excluded"]]
    gen = [r for r in live if r["generic_M"]]
    deg = [r for r in live if not r["generic_M"]]

    def stat(vals):
        v = np.array(vals, dtype=float)
        sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        return float(v.mean()), sd, sd / np.sqrt(len(v))

    print("\n=== PRE-REGISTERED CRITERION (generic M only) ===")
    print(f"generic-M cells               : {len(gen)}")
    print(f"corrected law exact           : {sum(r['corrected_law_exact'] for r in gen)}/{len(gen)}")
    print(f"paper law exact               : {sum(r['paper_law_exact'] for r in gen)}/{len(gen)}")
    m_, s_, e_ = stat([r["delta_dA_corrected"] for r in gen])
    print(f"paired delta d_A (corrected)  : mean {m_:+.4f}  std {s_:.4f}  se {e_:.4f}")
    m2, s2, e2 = stat([r["delta_surplus_paper"] for r in gen])
    print(f"paired delta surplus (paper)  : mean {m2:+.4f}  std {s2:.4f}  se {e2:.4f}")
    print("VERDICT ((dim A)^2 law):", "CONFIRMS" if all(r["corrected_law_exact"] for r in gen) else "KILLS")
    pw = [r for r in gen if r["pairwise_law_exact"] is not None]
    print(f"pairwise law exact            : {sum(r['pairwise_law_exact'] for r in pw)}/{len(pw)}")
    mixed = sorted({r['group'] for r in gen if r['multiplicity_regime'] == 'mixed'})
    print(f"mixed-multiplicity reps       : {mixed}  (closed forms fail here by construction)")

    ident = [r for r in live if r["M_family"] == "identity"]
    print("\n--- measurement only (not pass/fail) ---")
    print(f"degenerate/stabilised cells   : {len(deg)}")
    print(f"corrected law exact there     : {sum(r['corrected_law_exact'] for r in deg)}/{len(deg)}")
    print(f"M=I cells with d_A == dim A   : {sum(r['d_A_equals_dimA'] for r in ident)}/{len(ident)}")
    return rows


if __name__ == "__main__":
    main()
