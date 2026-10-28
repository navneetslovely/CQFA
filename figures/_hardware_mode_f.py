
from __future__ import annotations
import csv, os, shutil
import numpy as np
from figures.style import apply_style, new_fig, bold_ticks, thin_ticks, save, PALETTE

from cqfa.paths import ROOT as _ARCHIVE_ROOT

ROOT = str(_ARCHIVE_ROOT)
def load(tag):
    p = os.path.join(ROOT, "results", "ibm_campaign", tag, "analysis", "c8.csv")
    return list(csv.DictReader(open(p)))


def per_N(rows, mode):
    """mean peak_prob over shifts at each N for the given mode (+ min/max)."""
    Ns = sorted({int(r["N"]) for r in rows})
    out = {}
    for N in Ns:
        pk = [float(r["peak_prob"]) for r in rows if int(r["N"]) == N and r["mode"] == mode]
        out[N] = (np.mean(pk), np.min(pk), np.max(pk))
    return Ns, out


def main():
    apply_style()
    r1, r2 = load("ibm_modef"), load("ibm_modef2")
    Ns, F1 = per_N(r1, "F")
    _,  F2 = per_N(r2, "F")
    _,  B1 = per_N(r1, "B")
    _,  B2 = per_N(r2, "B")
    x = np.array(Ns, dtype=float)

    fig, ax = new_fig()
    # 1/N chance reference for Mode B
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, 1.0 / xx, "--", color=PALETTE["neutral"], lw=1.8, label=r"$1/N$ (blind)")

    def series(D, **kw):
        m = np.array([D[N][0] for N in Ns])
        lo = np.array([D[N][0] - D[N][1] for N in Ns])
        hi = np.array([D[N][2] - D[N][0] for N in Ns])
        ax.errorbar(x, m, yerr=[lo, hi], capsize=4, **kw)

    # Mode F: two device runs (filled = run-1, open = run-2)
    series(F1, marker="o", color=PALETTE["coherent"], mfc=PALETTE["coherent"],
           lw=2.6, label="Mode F (run 1)")
    series(F2, marker="o", color=PALETTE["coherent"], mfc="white", mew=2.0,
           lw=2.6, ls="--", label="Mode F (run 2)")
    # Mode B: pooled over runs (blind); no whisker (min=max=mean -> zero error)
    Bm = {N: ((lambda m: (m, m, m))(float(np.mean([B1[N][0], B2[N][0]])))) for N in Ns}
    series(Bm, marker="s", color=PALETTE["single_copy"], mfc=PALETTE["single_copy"],
           lw=2.6, label="Mode B (classical)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(N)) for N in Ns])
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(r"frame size $N=|G|$")
    ax.set_ylabel("peak probability")
    ax.legend(loc="center", fontsize=11, frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="0.75")
    bold_ticks(ax)
    pdf = save(fig, "f_modef_hw")
if __name__ == "__main__":
    main()
