"""Figure 5: cross-sector Mode F on IBM Heron.

Rebuilt from the recorded CSVs under results/. This script performs no
training and no simulation; it reads results and renders "f_modef_hw.pdf"
into figures/output/.
"""
from figures._hardware_mode_f import main as build


def main() -> None:
    """Render figure 5: cross-sector mode f on ibm heron."""
    build()


if __name__ == "__main__":
    main()
