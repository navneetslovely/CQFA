
from __future__ import annotations

import argparse
import csv
import os
import numpy as np

from . import spt_model as M
from . import string_family as F
from . import sample_complexity as SC


def _write_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _string_order_rows(sizes, spt_lambdas, trivial_lambdas):
    """string_order.csv: physics sanity rows across lambda for each n."""
    rows = []
    lams = sorted(set(list(spt_lambdas) + list(trivial_lambdas) + [0.5, 1.0, 1.5]))
    for n in sizes:
        for lam in lams:
            if abs(lam) < 1e-12:
                psi = M.cluster_state(n)
            else:
                psi = M.ground_state(n, lam)
            rows.append({
                "n": n, "lambda": lam,
                "string_order": M.string_order(psi, n),
                "local_z_mid": M.local_z_mid(psi, n),
                "z2xz2_commutator_max": M.z2xz2_commutator_max(n, lam),
            })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="SPT measurement-complexity separation")
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 8, 10, 12])
    ap.add_argument("--spt-lambdas", type=float, nargs="+", default=list(M.SPT_LAMBDAS))
    ap.add_argument("--trivial-lambdas", type=float, nargs="+", default=list(M.TRIVIAL_LAMBDAS))
    ap.add_argument("--task", default="optionA_stringfamily",
                    choices=["optionA_purity", "optionA_stringfamily", "optionB_unknown"])
    ap.add_argument("--arms", nargs="+",
                    default=["coherent", "shadow_nonlinear", "shadow_known", "local"])
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[16, 64, 256, 1024, 4096, 16384])
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--edge-pin", type=float, default=0.0)
    ap.add_argument("--out", default="results/spt_measurement_cost/")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    out = args.out
    os.makedirs(out, exist_ok=True)

    raw_rows, cstar_rows = [], []
    # cstar_by_arm[arm] = {n: mean} for ratio + slopes
    cstar_by_arm = {arm: {} for arm in args.arms}
    cstar_vals_by_arm = {arm: {} for arm in args.arms}

    for arm in args.arms:
        for n in args.sizes:
            raw_rows += SC.collect_raw(arm, n, args.budgets, args.epsilon, args.seeds,
                                       edge_pin=args.edge_pin)
            agg = SC.cstar_over_seeds(arm, n, args.budgets, args.epsilon, args.seeds,
                                      edge_pin=args.edge_pin, n_boot=args.bootstrap)
            cstar_by_arm[arm][n] = agg["cstar_mean"]
            cstar_vals_by_arm[arm][n] = agg["cstar_values"]
            cstar_rows.append({
                "arm": arm, "n": n, "task": args.task,
                "cstar_mean": agg["cstar_mean"], "cstar_std": agg["cstar_std"],
                "cstar_ci_lo": agg["cstar_ci_lo"], "cstar_ci_hi": agg["cstar_ci_hi"],
                "n_seeds": agg["n_seeds"], "n_finite": agg["n_finite"],
                "target_metric": "functional_error", "epsilon": args.epsilon,
            })
            print(f"{arm:18s} n={n:2d}  C*={agg['cstar_mean']:>8.0f}  "
                  f"CI=[{agg['cstar_ci_lo']:.0f},{agg['cstar_ci_hi']:.0f}]  "
                  f"(finite {agg['n_finite']}/{agg['n_seeds']})")

    # ratio rows: shadow_nonlinear / coherent (the headline separation)
    ratio_rows = []
    if "shadow_nonlinear" in args.arms and "coherent" in args.arms:
        for n in args.sizes:
            num = np.array(cstar_vals_by_arm["shadow_nonlinear"][n], dtype=float)
            den = np.array(cstar_vals_by_arm["coherent"][n], dtype=float)
            # per-seed ratio where both finite
            m = np.isfinite(num) & np.isfinite(den) & (den > 0)
            if m.sum() == 0:
                ratio_rows.append({"n": n, "task": args.task, "ratio_mean": float("nan"),
                                   "ratio_std": float("nan"), "ratio_ci_lo": float("nan"),
                                   "ratio_ci_hi": float("nan")})
                continue
            r = num[m] / den[m]
            rmean, rlo, rhi = SC.bootstrap_ci(r, n_boot=args.bootstrap, seed=n)
            ratio_rows.append({
                "n": n, "task": args.task, "ratio_mean": float(rmean),
                "ratio_std": float(r.std(ddof=0)), "ratio_ci_lo": rlo, "ratio_ci_hi": rhi,
            })

    string_rows = _string_order_rows(args.sizes, args.spt_lambdas, args.trivial_lambdas)

    _write_csv(raw_rows, os.path.join(out, "raw_estimates.csv"))
    _write_csv(cstar_rows, os.path.join(out, "cstar.csv"))
    _write_csv(ratio_rows, os.path.join(out, "ratio.csv"))
    _write_csv(string_rows, os.path.join(out, "string_order.csv"))

    # slopes (integrity summary)
    print("\nlog2(C*) vs n slopes (flat ~ 0, exponential > 0):")
    for arm in args.arms:
        ns = args.sizes
        cs = [cstar_by_arm[arm][n] for n in ns]
        print(f"  {arm:18s} slope = {SC.log_cstar_slope(ns, cs):.3f}   C* = {cs}")

    if not args.no_plots:
        try:
            from . import plots
            plots.make_all(out)
        except Exception as e:  # plotting is optional; never fail the run on it
            print(f"[plots skipped: {e}]")

    print(f"\nCSVs written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
