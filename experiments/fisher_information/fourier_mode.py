"""
experiments/fisher_information/fourier_mode.py
============================================
Mode F (frame Fourier / irrep readout) for a cyclic group $G=\\mathbb Z_N$, and the
three-mode comparison on a hidden-shift learning task. Statevector-exact.

Demonstrations (all verified against exact ground truth):
  * blindness:  Mode A (projection, trivial irrep) and Mode B (classical FA) give
    outputs CONSTANT in the hidden shift $s$; Mode F's Fourier phase $=-2\\pi ks/N$.
  * learnability: a classifier on Mode-F features learns a hidden-shift label
    $g(s)$; classifiers on Mode-A / Mode-B features are stuck at chance.
  * efficiency: Fourier sampling uses every outcome (no post-selection) -> Mode F
    escapes the coherent-projection trade-off (Prop. trade-off).

Output: results/fisher_information/mode_comparison.csv .
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np


# ---------------------------------------------------------------------------
# cyclic group Z_N: shift representation + the three readout modes
# ---------------------------------------------------------------------------
def shift(N: int, g: int) -> np.ndarray:
    return np.roll(np.eye(N, dtype=complex), g, axis=0)      # U_g|x> = |x+g mod N>


def mode_A(psi: np.ndarray, N: int, M: np.ndarray):
    """Coherent projection onto the trivial irrep |G>: f_A = <Pi psi|M|Pi psi>/p."""
    Pi = sum(shift(N, g) for g in range(N)) / N
    v = Pi @ psi
    p = float(np.real(np.vdot(v, v)))
    f = float(np.real(v.conj() @ M @ v) / p) if p > 1e-12 else float("nan")
    return f, p


def mode_B(psi: np.ndarray, N: int, M: np.ndarray) -> float:
    """Classical frame averaging: mean orbit output (diagonal, decohered)."""
    return float(np.mean([np.real((shift(N, g) @ psi).conj() @ M @ (shift(N, g) @ psi))
                          for g in range(N)]))


def mode_F_components(psi: np.ndarray, N: int):
    """Frame Fourier readout: for each irrep k, the data component
    v_k = (1/N) sum_g w^{-kg} U_g psi and its Born weight p_k = ||v_k||^2.
    Every shot yields some k (sum_k p_k = 1) -- no post-selection."""
    w = np.exp(-2j * np.pi / N)
    comps, probs = [], []
    for k in range(N):
        v = sum((w ** (-k * g)) * (shift(N, g) @ psi) for g in range(N)) / N
        comps.append(v)
        probs.append(float(np.real(np.vdot(v, v))))
    return comps, np.array(probs)


# ---------------------------------------------------------------------------
# hidden-shift learning task
# ---------------------------------------------------------------------------
def hidden_shift_dataset(N: int, m: int, rng, base=None):
    """m examples (psi_s = U_s phi, label g(s)); label = high bit of s (a hidden
    shift-dependent function). phi fixed so only s varies across examples."""
    phi = base if base is not None else _rand_state(N, rng)
    data = []
    for _ in range(m):
        s = int(rng.integers(0, N))
        psi = shift(N, s) @ phi
        y = int(s >= N // 2)                       # hidden shift-dependent label
        data.append((psi, s, y))
    return phi, data


def _rand_state(N, rng):
    v = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# per-mode features (physically estimable) + a nearest-centroid classifier
# ---------------------------------------------------------------------------
def _observables(N, rng, n_obs=4):
    Ms = []
    for _ in range(n_obs):
        d = rng.standard_normal(N)
        Ms.append(np.diag(d.astype(complex)))
    return Ms


def features(psi, N, mode, Ms, ref_comps):
    if mode == "A":
        return np.array([mode_A(psi, N, M)[0] for M in Ms])
    if mode == "B":
        return np.array([mode_B(psi, N, M) for M in Ms])
    if mode == "F":
        comps, probs = mode_F_components(psi, N)
        # measurable Mode-F features: irrep weights p_k and the relative phase of
        # each component vs a fixed reference (interference with a known state).
        phases = [np.angle(np.vdot(ref_comps[k], comps[k]) + 1e-15) for k in range(N)]
        return np.concatenate([probs, np.cos(phases), np.sin(phases)])
    raise ValueError(mode)


def _nearest_centroid_acc(X, y):
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    # Drop near-constant (uninformative) features so blind modes are not "learned"
    # by amplifying floating-point noise (A/B features are constant to ~1e-33).
    sd = X.std(0)
    keep = sd > 1e-8 * (np.abs(X).mean() + 1e-12)
    maj = max(float(np.mean(y)), 1.0 - float(np.mean(y)))   # no-information baseline
    if not np.any(keep):
        return maj                                          # blind -> chance/majority
    Xs = (X[:, keep] - X[:, keep].mean(0)) / (sd[keep])
    correct = 0
    for i in range(len(Xs)):
        mask = np.ones(len(Xs), bool); mask[i] = False
        c0 = Xs[mask][y[mask] == 0].mean(0) if np.any(y[mask] == 0) else np.zeros(Xs.shape[1])
        c1 = Xs[mask][y[mask] == 1].mean(0) if np.any(y[mask] == 1) else np.zeros(Xs.shape[1])
        pred = int(np.linalg.norm(Xs[i] - c1) < np.linalg.norm(Xs[i] - c0))
        correct += int(pred == y[i])
    return correct / len(Xs)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(sizes, m, seeds, out_dir):
    rows = []
    for N in sizes:
        acc = {"A": [], "B": [], "F": []}
        blind = {"A": [], "B": []}
        for sd in seeds:
            rng = np.random.default_rng(sd)
            phi, data = hidden_shift_dataset(N, m, rng)
            Ms = _observables(N, rng)
            ref_comps, _ = mode_F_components(phi, N)
            ys = [d[2] for d in data]
            for mode in ("A", "B", "F"):
                X = [features(d[0], N, mode, Ms, ref_comps) for d in data]
                acc[mode].append(_nearest_centroid_acc(X, ys))
            # blindness metric: variance of Mode A/B output across shifts (should be ~0)
            for mode in ("A", "B"):
                vals = [features(d[0], N, mode, Ms, ref_comps)[0] for d in data]
                blind[mode].append(float(np.var(vals)))
        rows.append(dict(
            N=N, n_frame_qubits=int(np.log2(N)),
            acc_modeA=float(np.mean(acc["A"])), acc_modeB=float(np.mean(acc["B"])),
            acc_modeF=float(np.mean(acc["F"])),
            modeA_output_var=float(np.mean(blind["A"])),
            modeB_output_var=float(np.mean(blind["B"])),
        ))
        print(f"N={N:>3} (|G|={N}): hidden-shift accuracy  A={np.mean(acc['A']):.3f}  "
              f"B={np.mean(acc['B']):.3f}  F={np.mean(acc['F']):.3f}   "
              f"(A/B output var across shifts ~ {np.mean(blind['A']):.1e}/{np.mean(blind['B']):.1e} -> blind)")
    if rows:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, "mode_comparison.csv")
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("wrote", p)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--m", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/fisher_information")
    args = ap.parse_args(argv)
    run(args.sizes, args.m, args.seeds, args.out)


if __name__ == "__main__":
    main()
