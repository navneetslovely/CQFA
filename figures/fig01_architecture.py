
from cqfa.paths import FIGURES
from figures._architecture import build


def main() -> str:
    """Render figure 1: the cqfa layer and return the output path."""
    out = FIGURES / "output" / "f0_architecture.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    return build(str(out))


if __name__ == "__main__":
    main()
