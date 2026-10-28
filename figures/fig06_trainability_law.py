"""Figure 6: gradient-variance scaling.

Rebuilt from the recorded CSVs under results/. This script performs no
training and no simulation; it reads results and renders "f8_trainability_law.pdf"
into figures/output/.
"""
from figures._paper_figures import fig_trainability_law


def main() -> None:
    """Render figure 6: gradient-variance scaling."""
    fig_trainability_law()


if __name__ == "__main__":
    main()
