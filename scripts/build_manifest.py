
from __future__ import annotations

import hashlib
import pathlib
import sys

from cqfa.paths import ROOT

MANIFEST = ROOT / "MANIFEST.sha256"
SKIP_DIRS = {"__pycache__", ".git", ".venv", "output", ".pytest_cache", ".ipynb_checkpoints"}
SKIP_NAMES = {"MANIFEST.sha256", ".DS_Store"}
# build artefacts: `pip install -e .` writes src/<name>.egg-info into the tree
SKIP_SUFFIXES = (".egg-info", ".dist-info")


def iter_files():
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        if SKIP_DIRS & set(p.parts) or p.name in SKIP_NAMES:
            continue
        if any(part.endswith(SKIP_SUFFIXES) for part in p.parts):
            continue
        yield p


def digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    lines, total = [], 0
    for p in iter_files():
        lines.append(f"{digest(p)}  {p.relative_to(ROOT).as_posix()}")
        total += p.stat().st_size
    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"  {len(lines)} files, {total / 1e6:.1f} MB -> {MANIFEST.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
