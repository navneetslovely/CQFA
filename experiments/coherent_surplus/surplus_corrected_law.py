
from __future__ import annotations

import csv
import os

import numpy as np

from experiments.coherent_surplus.surplus_span_ranks import (
    build_cases, make_M, herm_parts, span_rank, M_FAMILIES, SEEDS,
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
                             d_A_equals_dimA=None)
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
    print("VERDICT:", "CONFIRMS" if all(r["corrected_law_exact"] for r in gen) else "KILLS")

    ident = [r for r in live if r["M_family"] == "identity"]
    print("\n--- measurement only (not pass/fail) ---")
    print(f"degenerate/stabilised cells   : {len(deg)}")
    print(f"corrected law exact there     : {sum(r['corrected_law_exact'] for r in deg)}/{len(deg)}")
    print(f"M=I cells with d_A == dim A   : {sum(r['d_A_equals_dimA'] for r in ident)}/{len(ident)}")
    return rows


if __name__ == "__main__":
    main()
