
from __future__ import annotations

import csv
import os
import sys

import numpy as np


def fit(x, y_copies, label):
    y = np.log2(np.asarray(y_copies, dtype=float))
    x = np.asarray(x, dtype=float)
    alpha, beta = np.polyfit(x, y, 1)
    resid = y - (alpha * x + beta)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"{label}: copies ~ 2^({alpha:.3f} n + {beta:.3f})  = ({2**alpha:.3f})^n"
          f"   R^2={r2:.3f}  max|resid|={np.max(np.abs(resid)):.3f} bits"
          f"   fitted on n={[int(v) for v in x]}")
    return dict(arm=label, alpha_bits_per_qubit=round(float(alpha), 4),
                base_per_qubit=round(float(2 ** alpha), 4),
                intercept_log2=round(float(beta), 4),
                r2=round(float(r2), 4),
                max_abs_resid_bits=round(float(np.max(np.abs(resid))), 4),
                n_min=int(x.min()), n_max=int(x.max()), n_points=len(x))


def main():
    d = "results/two_copy_separation"
    src = os.path.join(d, "learning_separation_scaled.csv")
    rows = list(csv.DictReader(open(src)))
    clean = [r for r in rows if int(r["singlecopy_capped_runs"]) == 0
             and r["singlecopy_copies"] not in ("", "nan")]
    if len(clean) < 3:
        print("not enough uncapped rungs to fit"); sys.exit(1)
    ns = [int(r["n"]) for r in clean]
    out = [fit(ns, [float(r["singlecopy_copies"]) for r in clean], "single_copy_shadows"),
           fit(ns, [float(r["coherent_copies"]) for r in clean], "coherent_two_copy")]
    dropped = [int(r["n"]) for r in rows if r not in clean]
    if dropped:
        print(f"excluded from fit (capped -> lower bounds only): n={dropped}")
    p = os.path.join(d, "learning_separation_scaled_fit.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print("wrote", p)


if __name__ == "__main__":
    main()
