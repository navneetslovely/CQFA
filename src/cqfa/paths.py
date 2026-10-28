"""Filesystem locations used by the experiments, analysis and figure scripts.

Every path is resolved relative to the archive root, which is located by
walking up from this file. Nothing in the package depends on the working
directory or on any absolute path, so the archive can be unpacked anywhere.

Set ``CQFA_ROOT`` to override the archive root explicitly.
"""
from __future__ import annotations

import os
import pathlib

__all__ = [
    "ROOT",
    "SRC",
    "EXPERIMENTS",
    "ANALYSIS",
    "FIGURES",
    "FIGURES_PDF",
    "RESULTS",
    "RESULTS_RAW",
    "RESULTS_CURATED",
    "RESULTS_HARDWARE",
    "RUNS",
    "ensure",
]


def _find_root() -> pathlib.Path:
    override = os.environ.get("CQFA_ROOT")
    if override:
        return pathlib.Path(override).resolve()
    # src/cqfa/paths.py -> src/cqfa -> src -> archive root
    return pathlib.Path(__file__).resolve().parents[2]


ROOT = _find_root()

SRC = ROOT / "src"
EXPERIMENTS = ROOT / "experiments"
ANALYSIS = ROOT / "analysis"
FIGURES = ROOT / "figures"
FIGURES_PDF = FIGURES / "pdf"
RESULTS = ROOT / "results"
RESULTS_RAW = RESULTS / "raw"
RESULTS_CURATED = RESULTS / "curated"
RESULTS_HARDWARE = RESULTS / "hardware"
RUNS = ROOT / "runs"


def ensure(path: pathlib.Path) -> pathlib.Path:
    """Create ``path`` as a directory if it does not exist, and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
