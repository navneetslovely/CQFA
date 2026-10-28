
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from experiments.trainability_law.trainability_full import fit_loglog, round_to_unc, spearman


def _num(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _write(path, rows):
    if not rows:
        print("  (no rows for", path, ")")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"wrote {path} ({len(rows)} rows)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/trainability_law")
    ap.add_argument("--tag", default="trainability_full")
    args = ap.parse_args(argv)

    rows = load(os.path.join(args.indir, f"{args.tag}.csv"))
    gv = [r for r in rows if r["block"] == "grad_variance" and r["vacuous_excluded"] == "0"]
    opt = [r for r in rows if r["block"] == "optimisation" and r["vacuous_excluded"] == "0"]

    # =====================================================================
    # A. variance <-> optimisation link
    # =====================================================================
    out = []
    print("=== A. Does initial gradient variance predict optimisation success? ===")
    for scale in sorted({r["init_scale"] for r in opt}):
        sub = [r for r in opt if r["init_scale"] == scale]
        g = np.array([_num(r["init_grad_sq_mean"]) for r in sub])
        succ = np.array([_num(r["success_fraction"]) for r in sub])
        acc = np.array([_num(r["final_test_acc_mean"]) for r in sub])
        loss = np.array([_num(r["final_train_loss_mean"]) for r in sub])
        dim = np.array([_num(r["dim_hplus"]) for r in sub])
        dep = np.array([_num(r["depth"]) for r in sub])
        # iterations: only cells where EVERY restart reached the target are
        # unbiased; cells with capped restarts have a downward-biased mean.
        cap = np.array([_num(r["n_restarts_capped"], 0) for r in sub])
        it = np.array([_num(r["iters_to_target_mean"]) for r in sub])
        clean = (cap == 0) & np.isfinite(it)

        # Does the initial gradient collapse with dim H_+?  (it should)
        f_g = fit_loglog(dim, g)
        # Does the ITERATION COUNT grow as the gradient shrinks?
        f_it = fit_loglog(g[clean], it[clean]) if clean.sum() >= 3 else None
        f_it_dim = fit_loglog(dim[clean], it[clean]) if clean.sum() >= 3 else None

        rec = dict(
            init_scale=scale, n_cells=len(sub),
            n_cells_all_restarts_converged=int(clean.sum()),
            dim_hplus_min=float(dim.min()), dim_hplus_max=float(dim.max()),
            init_grad_sq_min=float(g.min()), init_grad_sq_max=float(g.max()),
            init_grad_sq_ratio_max_over_min=float(g.max() / g.min()),
            test_acc_min=float(acc.min()), test_acc_max=float(acc.max()),
            test_acc_mean=float(acc.mean()), test_acc_std=float(acc.std()),
            final_train_loss_min=float(loss.min()), final_train_loss_max=float(loss.max()),
            success_fraction_min=float(succ.min()), success_fraction_mean=float(succ.mean()),
            n_cells_success_1p0=int((succ == 1.0).sum()),
            spearman_grad_vs_success=spearman(g, succ),
            spearman_grad_vs_testacc=spearman(g, acc),
            spearman_grad_vs_finalloss=spearman(g, loss),
            spearman_dim_vs_testacc=spearman(dim, acc),
            spearman_depth_vs_testacc=spearman(dep, acc),
            slope_initgrad_vs_dimhplus=(f_g["slope"] if f_g else np.nan),
            slope_initgrad_vs_dimhplus_reported=(
                round_to_unc(f_g["slope"], f_g["ci_hi"] - f_g["slope"]) if f_g else ""),
            slope_iters_vs_initgrad=(f_it["slope"] if f_it else np.nan),
            slope_iters_vs_initgrad_reported=(
                round_to_unc(f_it["slope"], f_it["ci_hi"] - f_it["slope"]) if f_it else ""),
            r2_iters_vs_initgrad=(f_it["r2"] if f_it else np.nan),
            slope_iters_vs_dimhplus=(f_it_dim["slope"] if f_it_dim else np.nan),
            slope_iters_vs_dimhplus_reported=(
                round_to_unc(f_it_dim["slope"], f_it_dim["ci_hi"] - f_it_dim["slope"])
                if f_it_dim else ""),
            r2_iters_vs_dimhplus=(f_it_dim["r2"] if f_it_dim else np.nan),
        )
        out.append(rec)
        print(f"  init_scale={scale}: initial |g|^2 falls {rec['init_grad_sq_ratio_max_over_min']:.0f}x "
              f"over dim H_+ {rec['dim_hplus_min']:.0f}->{rec['dim_hplus_max']:.0f}")
        print(f"    test_acc stays {rec['test_acc_min']:.3f}-{rec['test_acc_max']:.3f} "
              f"(mean {rec['test_acc_mean']:.3f}); success_fraction=1.0 in "
              f"{rec['n_cells_success_1p0']}/{rec['n_cells']} cells")
        print(f"    iters_to_target ~ |g0|^{rec['slope_iters_vs_initgrad_reported']} "
              f"(R2={rec['r2_iters_vs_initgrad']:.3f}); "
              f"~ dimH+^{rec['slope_iters_vs_dimhplus_reported']} (R2={rec['r2_iters_vs_dimhplus']:.3f})")

    # per-(n,depth) detail so the paper can quote individual cells
    detail = []
    for r in opt:
        detail.append(dict(
            init_scale=r["init_scale"], n_qubits=r["n_qubits"], dim_hplus=r["dim_hplus"],
            depth=r["depth"], n_params=r["n_params"],
            init_grad_sq_mean=r["init_grad_sq_mean"],
            final_train_loss_mean=r["final_train_loss_mean"],
            final_test_acc_mean=r["final_test_acc_mean"],
            final_test_acc_std=r["final_test_acc_std"],
            success_fraction=r["success_fraction"],
            n_restarts=r["n_restarts"], n_restarts_capped=r["n_restarts_capped"],
            iters_to_target_mean=r["iters_to_target_mean"],
            iters_to_target_lower_bound=r["iters_to_target_lower_bound"],
            iters_mean_is_biased_by_capping=(1 if _num(r["n_restarts_capped"], 0) > 0 else 0),
        ))
    _write(os.path.join(args.indir, f"{args.tag}_optlink.csv"), out + [{}] + detail)

    # =====================================================================
    # B. CQFA advantage vs the STRONGEST baseline (integrity rule 4)
    # =====================================================================
    print("\n=== B. CQFA gradient advantage: strong vs weak baseline ===")
    adv = []
    for depth in sorted({int(r["depth"]) for r in gv}):
        for ik in ["aligned_gG", "haar_Hp", "gpure_max"]:
            for n in sorted({int(r["n_qubits"]) for r in gv}):
                c = [r for r in gv if r["arm"] == "cqfa_equivariant"
                     and int(r["depth"]) == depth and int(r["n_qubits"]) == n
                     and r["input_distribution"] == ik and r["observable"] == "dla_elem"
                     and r["param_position"] == "mid"]
                b = [r for r in gv if r["arm"] == "generic_hea"
                     and int(r["depth"]) == depth and int(r["n_qubits"]) == n
                     and r["observable"] == "proj_rank1_full" and r["param_position"] == "mid"]
                if not c or not b:
                    continue
                vc, vb = _num(c[0]["grad_var_mean"]), _num(b[0]["grad_var_mean"])
                weak = 1.0 / (4 ** n)
                adv.append(dict(
                    depth=depth, n_qubits=n, input_distribution=ik,
                    dim_hplus=c[0]["dim_hplus"], dim_full=c[0]["dim_full"],
                    dim_gG=c[0]["dim_gG"],
                    var_cqfa=vc,
                    var_generic_empirical_same_depth=vb,
                    var_generic_ragone_4n=weak,
                    ratio_vs_strong_baseline=vc / vb,
                    ratio_vs_weak_baseline=vc / weak,
                    generic_empirical_over_ragone=vb / weak,
                ))
    _write(os.path.join(args.indir, f"{args.tag}_advantage.csv"), adv)
    for ik in ["aligned_gG", "haar_Hp"]:
        s = [a for a in adv if a["depth"] == 4 and a["input_distribution"] == ik]
        if s:
            print(f"  D=4 {ik:11s}: ratio vs STRONG baseline "
                  f"{min(a['ratio_vs_strong_baseline'] for a in s):.1f}x .. "
                  f"{max(a['ratio_vs_strong_baseline'] for a in s):.1f}x "
                  f"(n={min(a['n_qubits'] for a in s)}..{max(a['n_qubits'] for a in s)})")
    gr = [a["generic_empirical_over_ragone"] for a in adv]
    print(f"  generic HEA empirical / Ragone-1/4^n = {np.mean(gr):.2f} +/- {np.std(gr):.2f} "
          f"(the weak baseline understates the real generic variance by this factor)")


if __name__ == "__main__":
    main()
