"""Supplementary Figure 2: measurement complexity.

Rebuilt from the recorded CSVs under results/. This script performs no
training and no simulation; it reads results and renders "f2b_exponential_separation.pdf"
into figures/output/.
"""
from figures._paper_figures import fig_exponential_separation


def main() -> None:
    """Render supplementary figure 2: measurement complexity."""
    fig_exponential_separation()


if __name__ == "__main__":
    main()
