
from __future__ import annotations

import csv
import os

import numpy as np

from experiments.coherent_surplus.surplus_span_ranks import (
    build_cases, make_M, herm_parts, span_rank, SEEDS,
)


def complex_rank(ops, tol=1e-9):
    A = np.array([o.reshape(-1) for o in ops])
    nrm = np.linalg.norm(A, axis=1, keepdims=True)
    keep = (nrm[:, 0] / nrm.max()) >= 1e-10
    A = A[keep] / nrm[keep]
    sv = np.linalg.svd(A, compute_uv=False)
    return int((sv > tol * sv[0]).sum())


def gram_rank(ops, tol=1e-9):
    mx = max(np.linalg.norm(x) for x in ops)
    keep = [o for o in ops if np.linalg.norm(o) / mx >= 1e-10]
    V = np.array([o.reshape(-1) / np.linalg.norm(o) for o in keep])
    Gm = V.conj() @ V.T                       # Hilbert-Schmidt Gram matrix (PSD)
    ev = np.linalg.eigvalsh(Gm)[::-1].real
    return int((ev > tol * ev[0]).sum())


def main(out_dir="results/coherent_surplus"):
    rows = []
    for case in build_cases():
        d, m, Us = case.dim, case.order, case.mats
        dimA = complex_rank(list(Us))          # dim of the group-algebra image
        for fam in ("gue_generic", "identity", "subgroup_averaged"):
            for seed in SEEDS:
                rng = np.random.default_rng(1000 * seed + 17)
                M, _, stab = make_M(fam, d, rng, Us)
                diag_raw = [Us[g].conj().T @ M @ Us[g] for g in range(m)]
                pair_raw = [Us[g].conj().T @ M @ Us[h] for g in range(m) for h in range(m)]
                dh, ph = [], []
                for C in diag_raw:
                    dh += herm_parts(C)
                for C in pair_raw:
                    ph += herm_parts(C)
                rows.append(dict(
                    group=case.label, order_G=m, dim_d=d, rep_type=case.rep_type,
                    M_family=fam, seed=seed, dim_group_algebra=dimA,
                    dimA_squared=dimA * dimA, stab_subgroup_order=stab,
                    dB_methodA_realvec=span_rank(dh)["rank"],
                    dB_methodB_complex=complex_rank(diag_raw),
                    dB_methodC_gram=gram_rank(diag_raw),
                    dA_methodA_realvec=span_rank(ph)["rank"],
                    dA_methodB_complex=complex_rank(pair_raw),
                    dA_methodC_gram=gram_rank(pair_raw),
                    bound_dA=min(m * m, d * d),
                    dA_equals_dimA_squared=None,
                ))
                r = rows[-1]
                r["methods_agree"] = (
                    r["dB_methodA_realvec"] == r["dB_methodB_complex"] == r["dB_methodC_gram"]
                    and r["dA_methodA_realvec"] == r["dA_methodB_complex"] == r["dA_methodC_gram"]
                )
                r["dA_equals_dimA_squared"] = (r["dA_methodA_realvec"] == dimA * dimA)

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "surplus_span_ranks_crosscheck.csv")
    assert not os.path.exists(p), f"refusing to overwrite {p}"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_dis = sum(1 for r in rows if not r["methods_agree"])
    print("wrote", p, f"({len(rows)} rows)")
    print("METHOD DISAGREEMENTS (must be 0):", n_dis)
    print("\ngue_generic (seed 0), dim(A) vs d_A vs bound:")
    print(f"{'group':>16}{'|G|':>5}{'d':>4}{'dimA':>6}{'dimA^2':>8}{'d_A':>6}{'min(|G|^2,d^2)':>16}")
    for r in rows:
        if r["M_family"] == "gue_generic" and r["seed"] == 0:
            print(f"{r['group']:>16}{r['order_G']:>5}{r['dim_d']:>4}{r['dim_group_algebra']:>6}"
                  f"{r['dimA_squared']:>8}{r['dA_methodA_realvec']:>6}{r['bound_dA']:>16}")
    return rows


if __name__ == "__main__":
    main()
