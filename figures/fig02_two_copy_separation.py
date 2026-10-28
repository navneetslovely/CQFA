"""Figure 2: copies required to reach the target accuracy.

Rebuilt from the recorded CSVs under results/. This script performs no
training and no simulation; it reads results and renders "f_fair.pdf"
into figures/output/.
"""
from figures._fair_and_noise import fig_fair


def main() -> None:
    """Render figure 2: copies required to reach the target accuracy."""
    fig_fair()


if __name__ == "__main__":
    main()
