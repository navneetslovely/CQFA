"""figures/scripts/make_paper_figs2.py
====================================
Regenerators for f_fair (line plot) and f_noise (heatmap), reproducing their
original plot TYPE and style. Only the reported problems are fixed:
  f_fair : y-axis label was cut; legend sat on top of the lines.
  f_noise: heatmap x-axis strength labels were crowded/overlapping.
Nothing else about the look is changed. Copies each PDF next to the manuscript.

Sources:
  f_fair  : results/two_copy_separation/{learning_separation,entanglement,overlap}.csv
  f_noise : results/robustness/{modef_allnoise,twocopy_allnoise}.csv
"""
from __future__ import annotations
import os, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figures.style import apply_style, new_fig, bold_ticks, thin_ticks, save, outline_fonts

from cqfa.paths import ROOT as _ARCHIVE_ROOT

ROOT = str(_ARCHIVE_ROOT)
R = lambda *p: os.path.join(ROOT, *p)

# original f_fair colours/markers (matplotlib-tab style), one series per task
FAIR = [("learning_separation", "purity", "n", "o", "-", "#C2379B"),
        ("entanglement", "entangle", "nA", "s", "--", "#2CA02C"),
        ("overlap", "overlap", "n", "^", ":", "#FF7F0E")]
COH = "#1F77B4"

# 10 CPTP channels, short labels in the original row order
CH_ORDER = ["amplitude_damping", "bit_flip", "bit_phase_flip", "coherent_x", "coherent_z",
            "dephasing", "depolarizing", "pauli_asym", "phase_damping", "thermal(nbar=0.1)"]
CH_SHORT = ["amp.damp", "bit-flip", "bit-phase", "coh-X", "coh-Z",
            "dephase", "depol", "Pauli-asym", "phase-damp", "thermal"]




def fig_fair():
    apply_style()
    fig, ax = new_fig()
    for fname, lab, ncol, mk, ls, col in FAIR:
        d = pd.read_csv(R("results", "two_copy_separation", f"{fname}.csv")).sort_values(ncol)
        ax.plot(d[ncol], d["singlecopy_copies"], marker=mk, ls=ls, color=col,
                label=f"single-copy ({lab})")
    # coherent 2-copy: flat, a few copies (line + diamond markers, as before)
    d0 = pd.read_csv(R("results", "two_copy_separation", "learning_separation.csv")).sort_values("n")
    ax.plot(d0["n"], d0["coherent_copies"], marker="D", ls="-", color=COH, label="coherent 2-copy")
    ax.set_yscale("log")
    ax.set_xlabel(r"system size $n$")
    ax.set_ylabel("copies to target")            # FIX: shorter label so it is not cut
    ax.set_ylim(2, 900)
    # FIX: legend moved into the empty band between the coherent line (bottom) and
    # the single-copy cluster (top), with a solid box so it does not sit on the curves.
    ax.legend(loc="center", bbox_to_anchor=(0.53, 0.34), fontsize=8, frameon=True,
              framealpha=0.96, facecolor="white", edgecolor="0.75")
    bold_ticks(ax); thin_ticks(ax, nx=5)
    save(fig, "f_fair")


def _pivot(df, value, transform=None):
    p = df.pivot_table(index="model", columns="strength", values=value, aggfunc="mean")
    p = p.reindex(CH_ORDER)
    return (transform(p) if transform else p)


def fig_noise():
    m = pd.read_csv(R("results", "robustness", "modef_allnoise.csv"))
    tc = pd.read_csv(R("results", "robustness", "twocopy_allnoise.csv"))
    acc = _pivot(m, "acc")
    ratio = _pivot(tc, "separation_ratio", transform=lambda p: np.log2(p))
    strengths = list(acc.columns)

    plt.rcParams.update({"axes.titlesize": 13})     # heatmaps use short panel titles
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    fig.subplots_adjust(wspace=0.55, bottom=0.24)

    def heat(ax, data, cmap, title, vmin, vmax):
        im = ax.imshow(data.to_numpy(dtype=float), aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax)
        ax.set_title(title, fontweight="bold")
        ax.set_yticks(range(len(CH_SHORT)))
        ax.set_yticklabels(CH_SHORT, fontsize=9)
        ax.set_xticks(range(len(strengths)))
        # FIX: rotate the strength labels so they no longer run together
        ax.set_xticklabels([f"{s:g}" for s in strengths], rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("noise strength", fontweight="bold", fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    heat(axes[0], acc, "viridis", "Mode F accuracy", 0.5, 1.0)
    heat(axes[1], ratio, "magma", r"2-copy $\log_2$(sep. ratio)", 0.0, 6.0)

    fig.savefig(R("figures", "output", "f_noise.pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(R("figures", "output", "f_noise.png"), bbox_inches="tight", pad_inches=0.08)
    # this figure bypasses save(), so outline its text explicitly (no Type 3, no Identity-H)
    outline_fonts(R("figures", "output", "f_noise.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_fair()
    fig_noise()
    print("done.")
