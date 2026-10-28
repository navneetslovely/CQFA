"""Supplementary Figure 3: ten channels by six strengths.

Rebuilt from the recorded CSVs under results/. This script performs no
training and no simulation; it reads results and renders "f_noise.pdf"
into figures/output/.
"""
from figures._fair_and_noise import fig_noise


def main() -> None:
    """Render supplementary figure 3: ten channels by six strengths."""
    fig_noise()


if __name__ == "__main__":
    main()
