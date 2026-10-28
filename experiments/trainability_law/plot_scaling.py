
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def plot_variance_vs_n(var_rows, outdir, mu_kind="early"):
    models = sorted({r["model"] for r in var_rows})
    plt.figure(figsize=(6, 4.2))
    for m in models:
        rows = sorted([r for r in var_rows
                       if r["model"] == m and r["mu_kind"] == mu_kind],
                      key=lambda r: int(r["n"]))
        ns = [int(r["n"]) for r in rows]
        V = [_f(r["var_mean"]) for r in rows]
        lo = [_f(r["var_ci_lo"]) for r in rows]
        hi = [_f(r["var_ci_hi"]) for r in rows]
        plt.plot(ns, V, "o-", label=m)
        plt.fill_between(ns, lo, hi, alpha=0.2)
    plt.yscale("log")
    plt.xlabel("number of data qubits n")
    plt.ylabel(r"Var$_\theta[\partial_\mu f]$  ($\mu$=early)")
    plt.title("gradient-variance vs qubit count")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "e5_variance_vs_n.pdf")
    plt.savefig(path)
    plt.close()
    return path


def plot_var_vs_invdim(var_rows, outdir, mu_kind="early"):
    plt.figure(figsize=(5.5, 4.2))
    have = False
    models = sorted({r["model"] for r in var_rows})
    for m in models:
        rows = [r for r in var_rows
                if r["model"] == m and r["mu_kind"] == mu_kind
                and not np.isnan(_f(r["var_pred"]))]
        # use var_pred's implied dim via var_pred; we plot empirical V vs pred
        V = [_f(r["var_mean"]) for r in rows]
        Vp = [_f(r["var_pred"]) for r in rows]
        if V and any(v > 0 for v in Vp):
            plt.loglog([max(vp, 1e-300) for vp in Vp],
                       [max(v, 1e-300) for v in V], "o", label=m)
            have = True
    if have:
        lim = plt.gca().get_xlim()
        plt.plot(lim, lim, "k--", alpha=0.4, label="y=x")
    plt.xlabel(r"Ragone prediction $P_g(\rho)P_g(O)/\dim g$")
    plt.ylabel(r"empirical Var$_\theta[\partial_\mu f]$")
    plt.title("empirical vs Ragone-law variance")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "e5_var_vs_invdim.pdf")
    plt.savefig(path)
    plt.close()
    return path


def plot_dla_dim(dla_rows, outdir):
    amb = sorted([r for r in dla_rows if r["which"] == "generic_hea"],
                 key=lambda r: int(r["n"]))
    res = sorted([r for r in dla_rows if r["which"] == "cqfa_restricted"],
                 key=lambda r: int(r["n"]))
    plt.figure(figsize=(5.5, 4.2))
    if amb:
        plt.plot([int(r["n"]) for r in amb],
                 [_f(r["dim_g_ambient"]) for r in amb],
                 "s-", label=r"generic HEA $\dim g = 4^n-1$")
    if res:
        plt.plot([int(r["n"]) for r in res],
                 [_f(r["dim_g_restricted"]) for r in res],
                 "o-", label=r"CQFA restricted $\dim g_G$")
    plt.yscale("log")
    plt.xlabel("number of data qubits n")
    plt.ylabel(r"$\dim g$")
    plt.title("dynamical-Lie-algebra dimension (the dial)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "e5_dla_dim.pdf")
    plt.savefig(path)
    plt.close()
    return path


def main():
    ap = argparse.ArgumentParser(description="Scaling figures")
    ap.add_argument("--variance", type=str,
                    default="experiments/trainability_law/out/variance.csv")
    ap.add_argument("--dla", type=str,
                    default="experiments/trainability_law/out/dla.csv")
    ap.add_argument("--outdir", type=str, default="figures/")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    made = []
    if os.path.exists(args.variance):
        vr = _read_csv(args.variance)
        made.append(plot_variance_vs_n(vr, args.outdir))
        made.append(plot_var_vs_invdim(vr, args.outdir))
    if os.path.exists(args.dla):
        dr = _read_csv(args.dla)
        made.append(plot_dla_dim(dr, args.outdir))
    for p in made:
        print(f"[plot] wrote {p}")


if __name__ == "__main__":
    main()
