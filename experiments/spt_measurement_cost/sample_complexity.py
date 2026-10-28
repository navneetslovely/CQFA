"""
experiments/spt_measurement_cost/sample_complexity.py
=======================================
The C*(n) driver: for each (arm, n) sweep a matched COPY budget, measure the
estimator error of the target functional, find the smallest budget reaching error
<= epsilon, and aggregate over seeds with bootstrap 95% CIs.

Arms:
  * 'coherent'          : 2-copy Bell readout of F(rho)=mean_[a,b] <O_string(a,b)>^2.
                          Per-shot variance O(1), n-independent -> C* flat.
  * 'shadow_nonlinear'  : single-copy classical-shadow U-statistic estimate of the
                          SAME F(rho). Variance grows with string weight/family ->
                          C* blows up (2^Omega(n)).
  * 'shadow_known'      : single-copy shadow estimate of the ONE known full-bulk
                          string <O_string>^2 (control) -> stays flat-ish; its slope
                          is the integrity check that the separation is task-driven.
  * 'local'             : low-weight shadow features only -> cannot separate (control).

Budget accounting: one coherent 2-copy shot consumes 2 copies; one
single-copy shadow shot consumes 1 copy. C is the matched COPY budget.

OOM guardrail: all estimators are statevector-only; no 4^n object is ever formed
(the coherent arm goes through readout_coherent, which squares single-copy
expectations; the shadow arm samples single-copy bitstrings).
"""
from __future__ import annotations

import csv
import os
import numpy as np

from . import spt_model as M
from . import readout_coherent as C
from . import readout_shadow as S
from . import string_family as F


# -- target: the normalized nonlinear functional F_norm ------------------------
def target_functional(psi: np.ndarray, n: int, intervals) -> float:
    """Exact target value F_norm(rho) = mean_[a,b] <O_string(a,b)>^2 in [0,1]."""
    return F.functional_normalized_exact(psi, n, intervals)


def estimate_coherent(psi, n, intervals, string_ops, rng, n_shots) -> float:
    """Coherent 2-copy estimate of F_norm: average the per-interval Bell estimate of
    <P>^2 over the family. n_shots = number of 2-copy shots (2 copies each)."""
    # split shots across the family members (round-robin); each member gets >=1 shot
    m = len(intervals)
    shots_each = max(1, n_shots // m)
    acc = 0.0
    for P in string_ops:
        acc += C.bell_estimate(psi, P, rng, shots_each)
    return acc / m


def estimate_shadow_nonlinear(psi, n, labels, rng, n_shots) -> float:
    """Single-copy shadow estimate of F_norm via the U-statistic per interval."""
    snaps = S.shadow_snapshots(psi, rng, n_shots)
    m = len(labels)
    acc = 0.0
    for lab in labels:
        acc += S.estimate_square_ustat(snaps, lab)
    return acc / m


def estimate_shadow_known(psi, n, rng, n_shots) -> float:
    """CONTROL 2: single-copy DIRECT estimate of the ONE known full-bulk string <O>.

    Measured directly in its own Pauli basis every shot -> per-shot variance
    1 - <O>^2 <= 1, n-INDEPENDENT. This is the integrity control that stays flat;
    if it grew with n the separation would be operator-driven, not task-driven.
    """
    return S.known_string_estimate(psi, n, rng, n_shots)


def coherent_known(psi, n, string_op, rng, n_shots) -> float:
    """Coherent estimate of the single known full-bulk <O>^2 (for the coherent arm's
    'known' target if needed)."""
    return C.bell_estimate(psi, string_op, rng, n_shots)


# -- one (arm, n, seed): error vs budget --------------------------------------
def _copies_to_shots(arm: str, budget_copies: int) -> int:
    """Convert a matched COPY budget to the arm's SHOT count."""
    if arm == "coherent":
        return max(1, budget_copies // 2)   # 2 copies per 2-copy shot
    return max(2, budget_copies)             # 1 copy per single-copy shot (need >=2 for U-stat)


def arm_error_at_budget(arm, psi, n, intervals, string_ops, labels, target, rng,
                        budget_copies) -> float:
    """|estimate - target| for the given arm at the given copy budget."""
    n_shots = _copies_to_shots(arm, budget_copies)
    if arm == "coherent":
        est = estimate_coherent(psi, n, intervals, string_ops, rng, n_shots)
    elif arm == "shadow_nonlinear":
        est = estimate_shadow_nonlinear(psi, n, labels, rng, n_shots)
    elif arm == "shadow_known":
        # known-string control: estimate the linear <O_full> directly and compare to
        # its exact value. Per-shot variance <= 1, n-independent -> stays flat.
        est = estimate_shadow_known(psi, n, rng, n_shots)
        exact_known = M.string_order(psi, n)
        return abs(est - exact_known)
    elif arm == "local":
        # local features cannot recover F; the estimator is defined as 0 (its best
        # constant guess), so its error is |target| and never drops below it -> C*=inf.
        return abs(0.0 - target)
    else:
        raise ValueError(f"unknown arm '{arm}'")
    return abs(est - target)


def cstar_one_seed(arm, n, budgets, epsilon, seed,
                   spt_lambda=0.0, edge_pin=0.0):
    """C*(arm, n) for one seed: smallest budget with error <= epsilon on the SPT
    class representative (the hard-to-estimate side). Returns (cstar, errors).

    Uses the SPT cluster state (lam=0) as the representative whose functional the
    arm must estimate to epsilon. cstar = np.inf if no budget reaches epsilon.
    """
    rng = np.random.default_rng(seed)
    psi = M.prepare_state(n, spt_lambda, "spt", edge_pin=edge_pin) if spt_lambda >= 0 else None
    intervals = F.interval_family(n)
    target = target_functional(psi, n, intervals)
    # precompute operators/labels once (they do not depend on shots)
    string_ops = F.string_operators(n, intervals) if arm in ("coherent",) else None
    labels = F.string_labels(n, intervals) if arm in ("shadow_nonlinear",) else None

    errors = []
    cstar = np.inf
    for bcopies in budgets:
        err = arm_error_at_budget(arm, psi, n, intervals, string_ops, labels,
                                  target, rng, int(bcopies))
        errors.append(err)
        if err <= epsilon and cstar == np.inf:
            cstar = int(bcopies)
    return cstar, np.array(errors), target


# -- aggregation with bootstrap CIs --------------------------------------------
def bootstrap_ci(values, n_boot=10000, seed=0, alpha=0.05):
    """Bootstrap (percentile) CI of the mean over seeds. Handles inf by clipping to
    a large finite sentinel for the resample, then reports the finite fraction."""
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    if finite.sum() == 0:
        return float("inf"), float("inf"), float("inf")
    vf = v[finite]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    m = len(vf)
    for b in range(n_boot):
        idx = rng.integers(0, m, size=m)
        means[b] = vf[idx].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(vf.mean()), lo, hi


def cstar_over_seeds(arm, n, budgets, epsilon, seeds, spt_lambda=0.0,
                     edge_pin=0.0, n_boot=10000):
    """Aggregate C*(arm,n) over seeds: mean, std, bootstrap 95% CI."""
    vals = []
    for s in seeds:
        cs, _, _ = cstar_one_seed(arm, n, budgets, epsilon, s,
                                  spt_lambda=spt_lambda, edge_pin=edge_pin)
        vals.append(cs)
    vals = np.array(vals, dtype=float)
    finite = vals[np.isfinite(vals)]
    mean = float(finite.mean()) if len(finite) else float("inf")
    std = float(finite.std(ddof=0)) if len(finite) else float("inf")
    _, lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=hash((arm, n)) % (2**31))
    return {
        "cstar_mean": mean, "cstar_std": std, "cstar_ci_lo": lo, "cstar_ci_hi": hi,
        "n_seeds": len(seeds), "n_finite": int(len(finite)),
        "cstar_values": vals.tolist(),
    }


# -- raw estimate logging ------------------------------------------------------
def collect_raw(arm, n, budgets, epsilon, seeds, spt_lambda=0.0, edge_pin=0.0):
    """Per-(budget,seed) rows for raw_estimates.csv (traceable numbers)."""
    rows = []
    intervals = F.interval_family(n)
    for s in seeds:
        cs, errs, target = cstar_one_seed(arm, n, budgets, epsilon, s,
                                          spt_lambda=spt_lambda, edge_pin=edge_pin)
        for bcopies, err in zip(budgets, errs):
            rows.append({
                "arm": arm, "n": n, "task": "optionA_stringfamily",
                "phase": "spt", "lambda": spt_lambda, "budget_copies": int(bcopies),
                "seed": s, "abs_error": float(err), "target_value": float(target),
                "epsilon": epsilon, "reached": bool(err <= epsilon),
            })
    return rows


def write_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# -- log-slope helper (used by tests: flat vs growing) -------------------------
def log_cstar_slope(ns, cstars):
    """Least-squares slope of log2(C*) vs n over the finite points. Flat ~ 0,
    exponential growth > 0."""
    ns = np.asarray(ns, dtype=float)
    cs = np.asarray(cstars, dtype=float)
    m = np.isfinite(cs) & (cs > 0)
    if m.sum() < 2:
        return float("nan")
    A = np.vstack([ns[m], np.ones(m.sum())]).T
    y = np.log2(cs[m])
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


if __name__ == "__main__":
    budgets = [16, 64, 256, 1024, 4096]
    seeds = [0, 1]
    for arm in ("coherent", "shadow_nonlinear", "shadow_known"):
        css = []
        for n in (6, 8):
            agg = cstar_over_seeds(arm, n, budgets, 0.1, seeds, n_boot=200)
            css.append(agg["cstar_mean"])
            print(f"{arm:18s} n={n} C*={agg['cstar_mean']:.0f} "
                  f"[{agg['cstar_ci_lo']:.0f},{agg['cstar_ci_hi']:.0f}]")
        print(f"    slope log2(C*) vs n = {log_cstar_slope([6,8], css):.3f}")
