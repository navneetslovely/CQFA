"""Figure 4: frame-readout head on the hidden-shift task.

Rebuilt from the recorded CSV. This script performs no training and no
simulation; it reads results and renders "f_modef.pdf" into figures/output/.

Source
------
results/fisher_information/mode_comparison.csv

One row per group order. Modes A and B are equal to the last recorded digit at
every size, so the two traces coincide and are drawn with offset line styles to
keep both visible.
"""
import csv

from cqfa.paths import RESULTS
from figures.style import PALETTE, apply_style, new_fig, save

CHANCE = "#8A8A8A"


def load(path=None):
    """Return (group orders, Mode-A acc, Mode-B acc, Mode-F acc)."""
    path = path or RESULTS / "fisher_information" / "mode_comparison.csv"
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    order = [int(r["N"]) for r in rows]
    a = [float(r["acc_modeA"]) for r in rows]
    b = [float(r["acc_modeB"]) for r in rows]
    f = [float(r["acc_modeF"]) for r in rows]
    return order, a, b, f


def main() -> str:
    """Render Figure 4 and return the output path."""
    apply_style()
    order, acc_a, acc_b, acc_f = load()
    x = list(range(len(order)))

    fig, ax = new_fig()
    ax.axhline(0.5, color=CHANCE, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.plot(x, acc_f, marker="D", ms=4.0, lw=1.6, color=PALETTE["cqfa"], zorder=4)
    ax.plot(x, acc_a, marker="o", ms=4.0, lw=1.4, ls="--",
            color=PALETTE["augmented"], zorder=3)
    ax.plot(x, acc_b, marker="s", ms=3.4, lw=1.4, ls=":",
            color=PALETTE["non_equivariant"], zorder=3)

    ax.text(2.45, 0.945, "Mode F", color=PALETTE["cqfa"], fontsize=9,
            fontweight="bold", ha="center", va="bottom")
    ax.text(1.5, 0.615, "Modes A, B", color=PALETTE["non_equivariant"],
            fontsize=9, fontweight="bold", ha="center", va="bottom")
    ax.text(-0.18, 0.455, "chance", color=CHANCE, fontsize=9,
            ha="left", va="center")

    ax.set_xticks(x)
    ax.set_xticklabels([str(o) for o in order])
    ax.set_xlim(-0.28, len(x) - 0.72)
    ax.set_ylim(0.42, 1.08)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel(r"group order $|G|$")
    ax.set_ylabel("accuracy")
    return save(fig, "f_modef")


if __name__ == "__main__":
    main()
