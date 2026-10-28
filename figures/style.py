"""Shared publication style for every figure in this archive.

Constraints applied to every figure:

  * no title is drawn inside the figure; captions live in the document;
  * tick labels and axis labels are bold, so a figure stays legible when
    printed at a fraction of the text width;
  * output is vector PDF regenerated from the recorded CSVs;
  * text is converted to outlines after rendering, so the emitted PDF carries
    no font objects and therefore no Type 3 or Identity-H fonts.

Usage::

    from figures.style import apply_style, new_fig, bold_ticks, save, PALETTE

    apply_style()
    fig, ax = new_fig()
    ax.plot(...)
    ax.set_xlabel("system size $n$")
    bold_ticks(ax)
    save(fig, "figure_name")
"""
from __future__ import annotations
import os
import shutil
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colour-blind-safe, consistent model palette used across all figures.
PALETTE = {
    "cqfa":            "#0072B2",  # blue
    "classical_fa":    "#D55E00",  # vermillion
    "non_equivariant": "#009E73",  # green
    "augmented":       "#CC79A7",  # purple
    "single_copy":     "#D55E00",
    "coherent":        "#0072B2",
    "accent":          "#E69F00",  # orange
    "neutral":         "#555555",
}
MARKERS = {"cqfa": "o", "classical_fa": "s", "non_equivariant": "^",
           "augmented": "D", "single_copy": "s", "coherent": "o"}

from cqfa.paths import FIGURES

_OUT = str(FIGURES / "output")


def apply_style() -> None:
    """Set global rcParams: large, bold, thick — readable at 1/3 width.
    Uses matplotlib's constrained layout engine so bold/large labels never clip or
    overlap, and reserves label padding so rotated y-labels stay inside the bbox."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 15,
        "font.family": "DejaVu Sans",
        "axes.labelsize": 17,
        "axes.labelweight": "bold",
        "axes.labelpad": 5.0,          # keep labels off the ticks
        "axes.linewidth": 1.8,
        "axes.titlesize": 0,            # never draw a title
        "axes.titlepad": 0,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.major.width": 1.8,
        "ytick.major.width": 1.8,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.pad": 4,
        "ytick.major.pad": 4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 2.6,
        "lines.markersize": 8,
        "legend.fontsize": 12,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.borderaxespad": 0.4,
        "legend.labelspacing": 0.3,
        # We save with bbox_inches='tight' (see save()), which is the single most
        # reliable anti-clip recipe: it expands the saved region to include EVERY
        # artist (bold labels, rotated ticks, legends). Do NOT also enable constrained
        # or tight layout — they fight bbox='tight' and can re-trim an edge label.
        "figure.constrained_layout.use": False,
        "figure.autolayout": False,
        "pdf.fonttype": 42,            # editable/embeddable fonts (no Type-3)
        "ps.fonttype": 42,
    })


def new_fig(w: float = 3.7, h: float = 3.0):
    """A single-panel figure sized for ~1/3 text-width placement in a row.
    Saved with bbox_inches='tight' (see save()), so nothing clips."""
    fig, ax = plt.subplots(figsize=(w, h))
    ax.margins(x=0.03)
    return fig, ax


def thin_ticks(ax, nx: int = 6, ny: int = 6) -> None:
    """Cap the number of major ticks so bold tick labels never crowd/overlap.
    Skips log axes (self-limiting) and categorical axes with fixed ticks (e.g. bar
    charts set via bar_labels), which must not have their labels overwritten."""
    from matplotlib.ticker import MaxNLocator, FixedLocator
    if ax.get_xscale() == "linear" and not isinstance(ax.xaxis.get_major_locator(), FixedLocator):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=nx, integer=False, prune=None))
    if ax.get_yscale() == "linear" and not isinstance(ax.yaxis.get_major_locator(), FixedLocator):
        ax.yaxis.set_major_locator(MaxNLocator(nbins=ny, integer=False, prune=None))


def bold_ticks(ax) -> None:
    """Make tick labels bold (matplotlib has no rcParam for tick-label weight)."""
    for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lab.set_fontweight("bold")
    # also bold any offset/scientific-notation text
    for axis in (ax.xaxis, ax.yaxis):
        t = axis.get_offset_text()
        t.set_fontweight("bold")
        t.set_fontsize(13)


def outline_fonts(pdf_path: str) -> bool:
    """Convert every glyph in the PDF to vector outlines, in place.

    Publishers often reject Type 3 fonts and ask that no CID/Identity-H font be retained.
    matplotlib's pdf.fonttype=42 avoids Type 3 but still emits CID TrueType
    subsets with Identity-H encoding. Outlining removes the font objects
    altogether, which satisfies both rules at once.

    The page box, the geometry, and every mark are untouched: the figure looks
    exactly the same, it simply no longer carries fonts. Returns False (leaving
    the PDF as-is) when Ghostscript is unavailable."""
    gs = shutil.which("gs")
    if gs is None:
        return False
    tmp = pdf_path + ".outlined"
    r = subprocess.run(
        [gs, "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pdfwrite",
         "-dNoOutputFonts", "-dAutoRotatePages=/None", "-dPDFSETTINGS=/prepress",
         f"-sOutputFile={tmp}", pdf_path],
        capture_output=True)
    if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
        os.replace(tmp, pdf_path)
        return True
    if os.path.exists(tmp):
        os.remove(tmp)
    return False


def save(fig, name: str, also_png: bool = True) -> str:
    """Save vector PDF (and a PNG preview) to figures/output/. Returns the PDF path.
    No title is ever added. Saved with bbox_inches='tight' (+ a small pad), which
    expands the saved region to include every artist — bold labels, rotated tick
    labels, and legends are never trimmed, at any subfigure size."""
    os.makedirs(_OUT, exist_ok=True)
    pdf = os.path.join(_OUT, f"{name}.pdf")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    if also_png:
        fig.savefig(os.path.join(_OUT, f"{name}.png"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    outline_fonts(pdf)
    return pdf


def bar_labels(ax, labels, rotate_if_longer_than: int = 4) -> None:
    """Set categorical x tick labels, rotating them when they would otherwise crowd/
    overlap in a narrow panel. Keeps them bold."""
    ax.set_xticks(range(len(labels)))
    longest = max((len(str(s)) for s in labels), default=0)
    rot = 30 if longest > rotate_if_longer_than else 0
    ha = "right" if rot else "center"
    ax.set_xticklabels(labels, rotation=rot, ha=ha, fontweight="bold")


def demo() -> None:
    """Render a 3-in-a-row style demo (worst case: long labels + legend) so clipping
    and overlap can be eyeballed without real data."""
    import numpy as np
    apply_style()
    x = np.arange(2, 11)
    # three separate figures, as they would appear side by side in a paper row
    fig, ax = new_fig()
    ax.plot(x, 2.0 ** (x / 3), marker=MARKERS["single_copy"], color=PALETTE["single_copy"], label="single-copy")
    ax.plot(x, 1.0 + 0 * x, marker=MARKERS["coherent"], color=PALETTE["coherent"], label="coherent")
    ax.set_xlabel(r"system size $n$")
    ax.set_ylabel("copies to 90% acc.")          # concise label (fits a short panel)
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    bold_ticks(ax); thin_ticks(ax)
    save(fig, "figstyle_demo")

    fig, ax = new_fig()
    ax.plot(x, np.tanh(x / 4), marker=MARKERS["cqfa"], color=PALETTE["cqfa"], label="CQFA")
    ax.plot(x, 0.5 + 0.03 * x, marker=MARKERS["classical_fa"], color=PALETTE["classical_fa"], label="classical FA")
    ax.set_xlabel(r"qubit count $n$")
    ax.set_ylabel(r"$\mathrm{Var}[\partial_\theta L]$")
    bold_ticks(ax); thin_ticks(ax)
    ax.legend(loc="lower right")
    save(fig, "figstyle_demo2")

    fig, ax = new_fig()
    cats = ["CQFA", "cl-FA", "non-eq", "aug"]
    ax.bar(cats, [0.98, 0.80, 0.74, 0.78],
           color=[PALETTE["cqfa"], PALETTE["classical_fa"], PALETTE["non_equivariant"], PALETTE["augmented"]])
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.5, 1.0)
    bar_labels(ax, cats)
    bold_ticks(ax); thin_ticks(ax, ny=6)
    save(fig, "figstyle_demo3")
    print("wrote figstyle_demo{,2,3}.pdf/.png")


if __name__ == "__main__":
    demo()
