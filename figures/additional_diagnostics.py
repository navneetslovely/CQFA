
from figures._paper_figures import (  # noqa: F401
    fig_measurement_complexity,
    fig_gradient_variance,
    fig_dial,
    fig_dla,
    fig_dlog,
    fig_hardware,
    fig_benchmark,
    fig_copy_fusion,
)

ALL = [
    fig_measurement_complexity,
    fig_gradient_variance,
    fig_dial,
    fig_dla,
    fig_dlog,
    fig_hardware,
    fig_benchmark,
    fig_copy_fusion,
]


def main() -> None:
    """Render every diagnostic figure."""
    for fn in ALL:
        fn()


if __name__ == "__main__":
    main()
