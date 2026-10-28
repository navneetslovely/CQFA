"""Figure 7: the Representation Dial.

Rebuilt from the recorded CSVs under results/. This script performs no
training and no simulation; it reads results and renders "f11_two_knob.pdf"
into figures/output/.
"""
from figures._paper_figures import fig_two_knob


def main() -> None:
    """Render figure 7: the representation dial."""
    fig_two_knob()


if __name__ == "__main__":
    main()
