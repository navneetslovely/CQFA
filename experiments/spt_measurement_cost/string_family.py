"""
experiments/spt_measurement_cost/string_family.py
===================================
The symmetry-allowed string family S_string that makes the separation PROVABLE
The task is deliberately NOT "estimate the one known string" (which is
single-copy easy); it is the nonlinear functional over an exponentially-large
family, OR discrimination against a hidden (unknown) string.

Option A (nonlinear functional, the unconditional headline):
    F(rho) = sum_{P in S_string} <P>^2 = Tr[ G (rho(x)rho) ],  G = sum_P P(x)P.
  S_string here = the Y-decorated cluster-string operators over all sub-intervals
  [a,b] of the chain (|S_string| grows ~ n^2 for the interval family, and the full
  symmetry-image family is 2^Omega(n)). F is large in the SPT phase (many strings
  ~ +/-1) and small in the trivial phase.

Option B (unknown-string discrimination, the physical narrative):
    given rho promised SPT-or-trivial with the string label hidden as one of many,
    decide the phase. Implemented by sampling a random interval per (seed, example).

We enumerate the sub-interval family (n^2 members) as the concrete, laptop-feasible
proxy for the exponential family. The scaling argument (2^Omega(n) single-copy cost)
is cited from HKP21/CCHL; the finite-n curve here is the empirical realization.
"""
from __future__ import annotations

import numpy as np

from . import spt_model as M
from . import readout_shadow as S


def interval_family(n: int, min_len: int = 1):
    """All bulk sub-intervals [a,b] with 1 <= a <= b <= n-2 and length >= min_len.

    Returns a list of (a, b) pairs. Size ~ (n-2)(n-1)/2 = O(n^2); this is the
    laptop-feasible proxy for the exponentially-large symmetric string family.
    """
    out = []
    for a in range(1, n - 1):
        for b in range(a, n - 1):
            if (b - a + 1) >= min_len:
                out.append((a, b))
    return out


def string_operators(n: int, intervals=None):
    """Dense O_string(a,b) matrices for each interval (used by the exact functional
    and the coherent Bell readout). NB: these are 2^n x 2^n (single-copy) -- NEVER
    2^(2n). Only construct for the sizes you sweep."""
    if intervals is None:
        intervals = interval_family(n)
    return [M.string_operator(n, a, b) for (a, b) in intervals]


def string_labels(n: int, intervals=None):
    """The Pauli-label dicts {site:Pauli} for each interval (for shadow estimation)."""
    if intervals is None:
        intervals = interval_family(n)
    return [S.string_pauli_label(n, a, b) for (a, b) in intervals]


def functional_exact(psi: np.ndarray, n: int, intervals=None) -> float:
    """Exact F(rho) = sum_{[a,b]} <O_string(a,b)>^2 (statevector-only)."""
    if intervals is None:
        intervals = interval_family(n)
    total = 0.0
    for (a, b) in intervals:
        e = M.string_order(psi, n, a, b)
        total += e * e
    return float(total)


def functional_normalized_exact(psi: np.ndarray, n: int, intervals=None) -> float:
    """F(rho) normalized by |S_string| so the target sits in [0,1] and epsilon is
    comparable across n. F_norm = mean_{[a,b]} <O_string(a,b)>^2."""
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
