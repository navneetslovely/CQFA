
import csv

from cqfa.paths import RESULTS
from figures.style import PALETTE, apply_style, new_fig, save

CHANCE = "#8A8A8A"


def load(path=None):
    """Return (primes, quantum mean, quantum std, best classical mean)."""
    path = path or RESULTS / "discrete_log" / "dlp_wide" / "dlp_wide_agg.csv"
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    primes = [int(r["prime"]) for r in rows]
    q = [float(r["quantum_dlog_mean"]) for r in rows]
    q_sd = [float(r["quantum_dlog_std"]) for r in rows]
    c = [float(r["best_classical_mean"]) for r in rows]
    return primes, q, q_sd, c


def main() -> str:
    """Render Figure 3 and return the output path."""
    apply_style()
    primes, q, q_sd, c = load()

    fig, ax = new_fig()
    ax.axhline(0.5, color=CHANCE, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.errorbar(primes, q, yerr=q_sd, marker="D", ms=3.0, lw=1.4, elinewidth=0.8,
                capsize=1.6, color=PALETTE["cqfa"], zorder=4)
    ax.plot(primes, c, marker="o", ms=3.0, lw=1.4,
            color=PALETTE["augmented"], zorder=3)

    ax.text(150, 0.905, "quantum", color=PALETTE["cqfa"], fontsize=9,
            fontweight="bold", ha="center", va="top")
    ax.text(150, 0.60, "best classical", color=PALETTE["augmented"], fontsize=9,
            fontweight="bold", ha="center", va="bottom")
    ax.text(20, 0.455, "chance", color=CHANCE, fontsize=9, ha="left", va="center")

    ax.set_xscale("log")
    ax.set_xlim(18, 5200)
    ax.set_ylim(0.42, 1.08)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel(r"prime $p$")
    ax.set_ylabel("CV accuracy")
    return save(fig, "f_dlp_wide")


if __name__ == "__main__":
    main()
