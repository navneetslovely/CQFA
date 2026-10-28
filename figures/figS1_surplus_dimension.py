"""Supplementary Figure 1: coherent-surplus dimension.

Rebuilt from the recorded CSVs under results/. This script performs no
training and no simulation; it reads results and renders "f9_surplus_dimension.pdf"
into figures/output/.
"""
from figures._paper_figures import fig_surplus


def main() -> None:
    """Render supplementary figure 1: coherent-surplus dimension."""
    fig_surplus()


if __name__ == "__main__":
    main()
