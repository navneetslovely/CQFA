
from __future__ import annotations

import csv
import os
import numpy as np

IN = "results/copy_fusion/informed_comparator_perseed.csv"
OUT = "results/copy_fusion/informed_comparator_pooled.csv"


def main():
    rows = list(csv.DictReader(open(IN)))
    out = []
    for task in ["A_linear_pauli_string", "B_nonlinear_purity"]:
        rs = [x for x in rows if x["task"] == task and x["vacuous_excluded"] == "0"]
        d = np.array([float(x["delta_informed_minus_coherent"]) for x in rs])
        se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
        ns = sorted({int(x["n"]) for x in rs})
        per_n = [float(np.mean([float(x["delta_informed_minus_coherent"])
                                for x in rs if int(x["n"]) == n])) for n in ns]
        n_pos = int(sum(1 for v in per_n if v > 0))

        if task == "A_linear_pauli_string":
            av = np.array([float(x["var_per_shot_agnostic"]) for x in rs])
            cf = np.array([float(x["var_agnostic_closed_form"]) for x in rs])
            agn_relerr = float(np.mean(np.abs(av - cf) / cf))
            agn_relerr_max = float(np.max(np.abs(av - cf) / cf))
            iv = np.array([float(x["var_per_shot_informed"]) for x in rs])
            icf = np.array([float(x["var_informed_closed_form"]) for x in rs])
            inf_abserr_max = float(np.max(np.abs(iv - icf)))
            fp = [float(x["shadow_fastpath_check_relerr"]) for x in rs
                  if x["shadow_fastpath_check_relerr"]]
            fp_mean, fp_max, fp_cells = float(np.mean(fp)), float(np.max(fp)), len(fp)
        else:
            agn_relerr = agn_relerr_max = inf_abserr_max = ""
            fp_mean = fp_max = ""
            fp_cells = 0

        out.append(dict(
            task=task, n_cells=len(d), sizes=";".join(str(n) for n in ns),
            pooled_delta_informed_minus_coherent_mean=float(np.mean(d)),
            pooled_delta_se=se, pooled_delta_two_se=2 * se,
            verdict_pooled=("SURVIVES" if float(np.mean(d)) > 2 * se else "KILLED"),
            n_sizes=len(ns), n_sizes_with_positive_delta=n_pos,
            per_size_mean_deltas=";".join(f"{v:.6f}" for v in per_n),
            agnostic_var_empirical_vs_closedform_relerr_mean=agn_relerr,
            agnostic_var_empirical_vs_closedform_relerr_max=agn_relerr_max,
            informed_var_empirical_vs_closedform_abserr_max=inf_abserr_max,
            shadow_fastpath_check_relerr_mean=fp_mean,
            shadow_fastpath_check_relerr_max=fp_max,
            shadow_fastpath_check_cells=fp_cells))

    if os.path.exists(OUT):
        raise SystemExit(f"refusing to overwrite existing results file: {OUT}")
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(f"{r['task']}: pooled delta={r['pooled_delta_informed_minus_coherent_mean']:+.6f} "
              f"2se={r['pooled_delta_two_se']:.6f} -> {r['verdict_pooled']} "
              f"({r['n_sizes_with_positive_delta']}/{r['n_sizes']} sizes positive)")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
