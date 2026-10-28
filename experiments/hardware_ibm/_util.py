"""
experiments.hardware_ibm._util
==========================
Small shared helpers: run manifest (env + git + argv), CSV writing, and a
deterministic bootstrap 95% CI (matches the project integrity rules).
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def _git_hash() -> str:
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "no-git"
    except Exception:
        return "no-git"


def package_versions() -> dict:
    v = {"python": sys.version.split()[0]}
    for mod in ("numpy", "qiskit", "qiskit_aer", "qiskit_ibm_runtime"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            v[mod] = "not-installed"
    return v


def write_manifest(out_dir: str, argv, seeds, extra=None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "manifest.json")
    manifest = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_hash": _git_hash(),
        "argv": list(argv),
        "seeds": list(seeds) if seeds is not None else None,
        "versions": package_versions(),
    }
    if extra:
        manifest.update(extra)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def write_csv(path: str, rows, fieldnames=None) -> str:
    if not rows:
        # still write a header-only file if fieldnames given
        if fieldnames:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        return path
    fieldnames = fieldnames or list(rows[0].keys())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    return path


# ---------------------------------------------------------------------------
# bootstrap 95% CI (deterministic)
# ---------------------------------------------------------------------------
def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=0):
    """Deterministic percentile bootstrap CI of the mean. Returns (lo, hi)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, size=values.size, replace=True).mean()
                      for _ in range(n_boot)])
    lo = float(np.percentile(means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(means, 100 * (1 + ci) / 2))
    return lo, hi
