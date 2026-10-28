
from __future__ import annotations

import numpy as np

from . import spt_model as M
from . import readout_shadow as S


def interval_family(n: int, min_len: int = 1):
  
    out = []
    for a in range(1, n - 1):
        for b in range(a, n - 1):
            if (b - a + 1) >= min_len:
                out.append((a, b))
    return out


def string_operators(n: int, intervals=None):

    if intervals is None:
        intervals = interval_family(n)
    return [M.string_operator(n, a, b) for (a, b) in intervals]


def string_labels(n: int, intervals=None):
 
    if intervals is None:
        intervals = interval_family(n)
    return [S.string_pauli_label(n, a, b) for (a, b) in intervals]


def functional_exact(psi: np.ndarray, n: int, intervals=None) -> float:
   
    if intervals is None:
        intervals = interval_family(n)
    total = 0.0
    for (a, b) in intervals:
        e = M.string_order(psi, n, a, b)
        total += e * e
    return float(total)


def functional_normalized_exact(psi: np.ndarray, n: int, intervals=None) -> float:
   
    if intervals is None:
        intervals = interval_family(n)
    return functional_exact(psi, n, intervals) / max(1, len(intervals))


def random_interval(n: int, rng: np.random.Generator, min_len: int = 1):
    """Option B: draw one hidden interval [a,b] uniformly from the family."""
    fam = interval_family(n, min_len)
    idx = int(rng.integers(0, len(fam)))
    return fam[idx]


if __name__ == "__main__":
    for n in (6, 8, 10):
        fam = interval_family(n)
        psi_spt = M.cluster_state(n)
        psi_tri = M.ground_state(n, 2.5)
        print(f"n={n}: |S_string|={len(fam):3d}  "
              f"F_norm(SPT)={functional_normalized_exact(psi_spt, n):.4f}  "
              f"F_norm(trivial)={functional_normalized_exact(psi_tri, n):.4f}")
