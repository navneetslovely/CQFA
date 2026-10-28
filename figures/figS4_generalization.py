"""Supplementary Figure 4: sample efficiency.

Rebuilt from the recorded CSVs under results/. This script performs no
training and no simulation; it reads results and renders "f10_generalization.pdf"
into figures/output/.
"""
from figures._paper_figures import fig_generalization


def main() -> None:
    """Render supplementary figure 4: sample efficiency."""
    fig_generalization()


if __name__ == "__main__":
    main()
