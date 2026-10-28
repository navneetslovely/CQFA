
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


def plot_dial(points, scaling, outdir):
    plt.figure(figsize=(6.2, 5.4))
    ax = plt.gca()

    # quadrant background from the median split
    Ts = [_f(r["T_scalar_at_nmax"]) for r in scaling]
    Ts = [t for t in Ts if np.isfinite(t)]
    Hs = [_f(r["H_frame_entropy_at_nmax"]) * (1 if int(r["H_escape"]) else 0.3)
          for r in scaling]
    t_mid = np.median(Ts) if Ts else 0.0
    h_mid = np.median(Hs) if Hs else 0.5

    ax.axvline(h_mid, color="gray", lw=0.8, ls="--")
    ax.axhline(t_mid, color="gray", lw=0.8, ls="--")

    # scatter model endpoints
    for r in scaling:
        T = _f(r["T_scalar_at_nmax"])
        esc = int(r["H_escape"])
        H = _f(r["H_frame_entropy_at_nmax"]) * (1 if esc else 0.3)
        if not np.isfinite(T):
            T = ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -10
        ax.scatter([H], [T], s=140, zorder=5)
        ax.annotate(r["model"], (H, T), textcoords="offset points",
                    xytext=(8, 6), fontsize=9)

    # n-indexed trajectories per model
    models = sorted({r["model"] for r in points})
    for m in models:
        rows = sorted([r for r in points if r["model"] == m],
                      key=lambda r: int(r["n"]))
        ns = sorted({int(r["n"]) for r in rows})
        traj_T, traj_H = [], []
        for n in ns:
            sub = [r for r in rows if int(r["n"]) == n]
            vlb = np.mean([_f(r["T_var_lb"]) for r in sub])
            T = np.log10(vlb) if vlb > 0 else -3.0 * n
            esc = int(round(np.mean([int(r["H_escape"]) for r in sub])))
            H = np.mean([_f(r["H_frame_entropy"]) for r in sub]) * (1 if esc else 0.3)
            traj_T.append(T)
            traj_H.append(H)
        ax.plot(traj_H, traj_T, "-", alpha=0.5)

    ax.set_xlabel(r"hardness / non-triviality  $H$  (frame entropy $\times$ escape)")
    ax.set_ylabel(r"trainability  $T=\log_{10}\mathrm{Var}_{lb}$")
    ax.set_title("Certified Representation Dial")
    # quadrant labels
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    ax.text(xr, yt, "trainable & hard", ha="right", va="top", fontsize=8,
            color="green", alpha=0.7)
    ax.text(xl, yt, "trainable & simulable", ha="left", va="top", fontsize=8,
            alpha=0.6)
    ax.text(xr, yb, "hard & untrainable", ha="right", va="bottom", fontsize=8,
            alpha=0.6)
    ax.text(xl, yb, "untrainable & simulable", ha="left", va="bottom",
            fontsize=8, alpha=0.6)
    plt.tight_layout()
    path = os.path.join(outdir, "e5_representation_dial.pdf")
    plt.savefig(path)
    plt.close()
    return path


def main():
    ap = argparse.ArgumentParser(description="Representation-Dial figure")
    ap.add_argument("--points", type=str,
                    default="experiments/trainability_law/out/certificate_points.csv")
    ap.add_argument("--scaling", type=str,
                    default="experiments/trainability_law/out/certificate_scaling.csv")
    ap.add_argument("--outdir", type=str, default="figures/")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if os.path.exists(args.points) and os.path.exists(args.scaling):
        p = plot_dial(_read_csv(args.points), _read_csv(args.scaling), args.outdir)
        print(f"[dial] wrote {p}")
    else:
        print("[dial] missing input CSVs; run run_certificate first")


if __name__ == "__main__":
    main()
