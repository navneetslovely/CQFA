"""
experiments/two_copy_separation/learning_separation_scaled.py
=================================================================
SCALED version of learning_separation.py: same task, same learners, same
estimators -- but engineered so the single-copy arm can be pushed well past n=5.

What changed (algorithmics only -- the ESTIMATORS ARE IDENTICAL):

 1. Shadow Gram in closed form.  The single-copy purity U-statistic needs
    Tr(S_i S_j) for all snapshot pairs, where S_i = ⊗_q (3|b,s><b,s| - I).
    The original code built dense d x d snapshot matrices (O(C^2 d^2) work and
    O(C d^2) memory) -- hopeless past n≈6.  But the trace factorises per qubit:

        Tr(S_i S_j) = Π_q [ 9 |<v_iq|v_jq>|^2 - 4 ]
                    = 5^a * (-4)^b * (1/2)^c

    with a = #qubits where (basis,outcome) agree, b = #qubits where the basis
    agrees but the outcome differs, c = #qubits with different bases (a+b+c = n).
    a and b+a come from two one-hot Gram matmuls, so the whole C x C Gram is two
    BLAS calls + elementwise work: O(C^2 n) flops, no d^2 anywhere.
    Verified against the original dense implementation to <1e-6 (--verify).

 2. Snapshot sampling is batched.  A Haar state is rotated into B different random
    product Pauli eigenbases at once with n batched 2x2 einsums, then B outcomes are
    drawn by one cumsum+searchsorted.  Maximally mixed states need no simulation at
    all (every product-basis outcome is uniform).

 3. Exponential (doubling) search over the copy budget with SNAPSHOT AND GRAM REUSE:
    the ladder is nested, so the Gram block for budget C contains the block for C/2.
    Only the new L-shaped region is ever computed, so reaching budget C costs
    ~0.67 C^2 total instead of re-scanning.

Output: results/two_copy_separation/learning_separation_scaled.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time

import numpy as np

# --------------------------------------------------------------------------
# single-qubit Pauli eigenbases (columns = |b,0>, |b,1>) -- same convention as
# learning_separation.py
# --------------------------------------------------------------------------
_PAULI = {
    0: np.array([[0, 1], [1, 0]], dtype=complex),
    1: np.array([[0, -1j], [1j, 0]], dtype=complex),
    2: np.array([[1, 0], [0, -1]], dtype=complex),
}
_EIGVEC = {b: np.linalg.eigh(_PAULI[b])[1] for b in (0, 1, 2)}
# rows of _ROT[b] are <b,0| and <b,1|
_ROT = np.stack([_EIGVEC[b].conj().T for b in (0, 1, 2)])          # (3,2,2)


# --------------------------------------------------------------------------
# state ensembles
# --------------------------------------------------------------------------
def haar_vec(n: int, rng) -> np.ndarray:
    d = 2 ** n
    v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    return v / np.linalg.norm(v)


class State:
    """A state held in the cheapest faithful form.

    kind='pure'      : one state vector, purity 1.
    kind='maxmixed'  : I/d, purity 2^-n; product-basis outcomes are uniform bits.
    kind='rankr'     : uniform mixture of r Haar vectors (kept as vectors).
    """

    def __init__(self, kind, n, vecs=None):
        self.kind, self.n, self.vecs = kind, n, vecs
        if kind == "maxmixed":
            self.purity = 2.0 ** (-n)
        elif kind == "pure":
            self.purity = 1.0
        else:
            V = np.stack(vecs)                       # (r,d)
            r = V.shape[0]
            self.purity = float(np.sum(np.abs(V.conj() @ V.T) ** 2) / r ** 2)


def make_dataset(n, m, rng, mixed_rank=None):
    """Balanced (state, label) list; label 1 = pure, 0 = mixed."""
    data = []
    for _ in range(m // 2):
        data.append((State("pure", n, [haar_vec(n, rng)]), 1))
        if mixed_rank is None:
            data.append((State("maxmixed", n), 0))
        else:
            data.append((State("rankr", n,
                               [haar_vec(n, rng) for _ in range(mixed_rank)]), 0))
    rng.shuffle(data)
    return data


# --------------------------------------------------------------------------
# batched single-copy random-Pauli snapshots
# --------------------------------------------------------------------------
def _sample_outcomes_from_vecs(V, bases, rng, chunk=256):
    """V: (B,d) state vectors (one per snapshot). bases: (B,n).
    Rotate each vector into its own product eigenbasis, Born-sample one index."""
    B, d = V.shape
    n = bases.shape[1]
    idx_out = np.empty(B, dtype=np.int64)
    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        psi = V[s:e].astype(np.complex128, copy=True)
        b = e - s
        for q in range(n):
            M = _ROT[bases[s:e, q]]                                   # (b,2,2)
            t = psi.reshape(b, 2 ** q, 2, 2 ** (n - 1 - q))
            psi = np.einsum("bij,bljr->blir", M, t, optimize=True).reshape(b, d)
        p = np.abs(psi) ** 2
        p /= p.sum(axis=1, keepdims=True)
        cdf = np.cumsum(p, axis=1)
        u = rng.random((b, 1))
        idx_out[s:e] = np.minimum((cdf < u).sum(axis=1), d - 1)
    return idx_out


def sample_snapshots(state: State, C: int, rng):
    """Return (bases, outcomes) each (C,n) int8 for C fresh single-copy snapshots."""
    n = state.n
    bases = rng.integers(0, 3, size=(C, n)).astype(np.int8)
    if state.kind == "maxmixed":
        outs = rng.integers(0, 2, size=(C, n)).astype(np.int8)
        return bases, outs
    if state.kind == "pure":
        V = np.broadcast_to(state.vecs[0], (C, 2 ** n))
    else:
        r = len(state.vecs)
        pick = rng.integers(0, r, size=C)
        V = np.stack(state.vecs)[pick]
    idx = _sample_outcomes_from_vecs(np.ascontiguousarray(V), bases, rng)
    shifts = np.arange(n - 1, -1, -1)
    outs = ((idx[:, None] >> shifts[None, :]) & 1).astype(np.int8)
    return bases, outs


# --------------------------------------------------------------------------
# closed-form snapshot Gram:  Tr(S_i S_j) = 5^a (-4)^b (1/2)^c
# --------------------------------------------------------------------------
_L5, _L4, _L2 = math.log(5.0), math.log(4.0), math.log(2.0)


def _onehot(bases, outs, n):
    """(A,K): A one-hot over (qubit,basis) [3n cols], K over (qubit,basis,outcome) [6n]."""
    C = bases.shape[0]
    qs = np.arange(n)
    A = np.zeros((C, 3 * n), dtype=np.float32)
    K = np.zeros((C, 6 * n), dtype=np.float32)
    A[np.arange(C)[:, None], (3 * qs)[None, :] + bases] = 1.0
    K[np.arange(C)[:, None], (6 * qs)[None, :] + 2 * bases + outs] = 1.0
    return A, K


def _gram_block(Ai, Ki, Aj, Kj, n):
    """Tr(S_i S_j) for row-set i vs column-set j (dense; used by --verify)."""
    SB = Ai @ Aj.T                                   # # qubits with same basis
    SF = Ki @ Kj.T                                   # # qubits with same basis+outcome
    a = SF
    b = SB - SF
    c = n - SB
    mag = np.exp(a * _L5 + b * _L4 - c * _L2, dtype=np.float32)
    sign = 1.0 - 2.0 * (np.mod(b, 2.0))
    return mag * sign


def _gram_block_sum(Ai, Ki, Aj, Kj, n, chunk=2048):
    """sum_{i,j} Tr(S_i S_j) over a row-set x column-set, computed in row chunks so
    peak memory is O(chunk * n_cols) instead of O(n_rows * n_cols).  Accumulated in
    float64 because the summands span 5^n in magnitude with alternating sign."""
    tot = 0.0
    for s in range(0, Ai.shape[0], chunk):
        e = min(s + chunk, Ai.shape[0])
        tot += float(_gram_block(Ai[s:e], Ki[s:e], Aj, Kj, n).astype(np.float64).sum())
    return tot


class ShadowAccumulator:
    """Grows snapshots along a nested budget ladder, reusing the Gram.

    S(C) = sum_{i,j < C} Tr(S_i S_j).  Purity estimate = (S(C) - C*5^n)/(C(C-1))."""

    def __init__(self, state: State, rng):
        self.state, self.rng, self.n = state, rng, state.n
        self.bases = np.zeros((0, state.n), dtype=np.int8)
        self.outs = np.zeros((0, state.n), dtype=np.int8)
        self.A = np.zeros((0, 3 * state.n), dtype=np.float32)
        self.K = np.zeros((0, 6 * state.n), dtype=np.float32)
        self.S = 0.0
        self.C = 0

    def grow_to(self, C: int) -> float:
        if C > self.C:
            nb, no = sample_snapshots(self.state, C - self.C, self.rng)
            An, Kn = _onehot(nb, no, self.n)
            # new rows vs old cols (x2 by symmetry) + new rows vs new rows
            if self.C > 0:
                self.S += 2.0 * _gram_block_sum(An, Kn, self.A, self.K, self.n)
            self.S += _gram_block_sum(An, Kn, An, Kn, self.n)
            self.bases = np.concatenate([self.bases, nb])
            self.outs = np.concatenate([self.outs, no])
            self.A = np.concatenate([self.A, An])
            self.K = np.concatenate([self.K, Kn])
            self.C = C
        return (self.S - self.C * 5.0 ** self.n) / (self.C * (self.C - 1))


# --------------------------------------------------------------------------
# coherent 2-copy reader (destructive SWAP; per-shot +/-1, mean = purity)
# --------------------------------------------------------------------------
def coherent_feature(state: State, C: int, rng) -> float:
    p_plus = 0.5 * (1.0 + state.purity)
    return float(np.mean(np.where(rng.random(C) < p_plus, 1.0, -1.0)))


# --------------------------------------------------------------------------
# trained 1-D threshold classifier + copies-to-accuracy
# --------------------------------------------------------------------------
def _accuracy_at_threshold(feats, labels):
    feats = np.asarray(feats, dtype=float)
    labels = np.asarray(labels)
    order = np.argsort(feats)
    fs, ys = feats[order], labels[order]
    best = 0.5
    for t in (fs[:-1] + fs[1:]) / 2:
        best = max(best, float(np.mean((fs > t) == ys)), float(np.mean((fs <= t) == ys)))
    return best


def coherent_copies(data, rng, ladder, target):
    for C in ladder:
        feats = [coherent_feature(st, C, rng) for st, _ in data]
        if _accuracy_at_threshold(feats, [y for _, y in data]) >= target:
            return C, C
    return None, ladder[-1]


def singlecopy_copies(data, rng, ladder, target, budget_s=None, verbose=True):
    accs = ShadowAccumulator_list = [ShadowAccumulator(st, rng) for st, _ in data]
    labels = [y for _, y in data]
    t0 = time.time()
    reached, last = None, ladder[0]
    for C in ladder:
        feats = [acc.grow_to(C) for acc in accs]
        a = _accuracy_at_threshold(feats, labels)
        last = C
        if verbose:
            print(f"      C={C:>6d}  acc={a:.3f}  ({time.time()-t0:.1f}s)", flush=True)
        if a >= target:
            reached = C
            break
        if budget_s is not None and time.time() - t0 > budget_s:
            if verbose:
                print(f"      [time cap {budget_s}s hit at C={C}]", flush=True)
            break
    return reached, last


# --------------------------------------------------------------------------
# verification: closed-form Gram == dense reference
# --------------------------------------------------------------------------
def verify():
    from experiments.two_copy_separation.learning_separation import (
        _snapshot_matrix, shadow_purity_estimate, haar_pure)
    rng = np.random.default_rng(7)
    worst = 0.0
    for n in (1, 2, 3, 4):
        bases = rng.integers(0, 3, size=(12, n)).astype(np.int8)
        outs = rng.integers(0, 2, size=(12, n)).astype(np.int8)
        A, K = _onehot(bases, outs, n)
        fast = _gram_block(A, K, A, K, n)
        dense = np.zeros((12, 12))
        mats = [_snapshot_matrix(bases[i], outs[i], n) for i in range(12)]
        for i in range(12):
            for j in range(12):
                dense[i, j] = np.real(np.trace(mats[i] @ mats[j]))
        rel = np.max(np.abs(fast - dense) / np.maximum(np.abs(dense), 1.0))
        worst = max(worst, float(rel))
        print(f"  n={n}: max rel |fast-dense| = {rel:.3e}")
    # unbiasedness: shadow estimator mean -> true purity
    for n in (2, 3):
        v = haar_vec(n, rng)
        st = State("pure", n, [v])
        ests = []
        for t in range(60):
            ests.append(ShadowAccumulator(st, np.random.default_rng(1000 + t)).grow_to(64))
        print(f"  n={n} pure: shadow purity mean={np.mean(ests):.4f} "
              f"+/- {np.std(ests)/np.sqrt(len(ests)):.4f} (true 1.0)")
        st = State("maxmixed", n)
        ests = [ShadowAccumulator(st, np.random.default_rng(2000 + t)).grow_to(64)
                for t in range(60)]
        print(f"  n={n} I/d : shadow purity mean={np.mean(ests):.4f} "
              f"+/- {np.std(ests)/np.sqrt(len(ests)):.4f} (true {2.0**-n:.4f})")
    print(f"GRAM VERIFY worst relative error = {worst:.3e}")
    return worst


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def run(sizes, m, seeds, out_dir, mixed_rank=None, target=0.9,
        max_copies=1 << 16, time_cap=180.0):
    rows = []
    coh_ladder = [2, 4, 8, 16, 32, 64, 128]
    C = 4
    sc_ladder = []
    while C <= max_copies:
        sc_ladder.append(C)
        C *= 2
    for n in sizes:
        coh_c, sc_c, capped = [], [], 0
        t_n = time.time()
        for sd in seeds:
            rng = np.random.default_rng(sd)
            data = make_dataset(n, m, rng, mixed_rank=mixed_rank)
            c_coh, _ = coherent_copies(data, rng, coh_ladder, target)
            print(f"    n={n} seed={sd}: coherent -> {c_coh}", flush=True)
            c_sc, last = singlecopy_copies(data, rng, sc_ladder, target,
                                           budget_s=time_cap)
            if c_sc is None:
                capped += 1
                sc_c.append(np.nan)
            else:
                sc_c.append(c_sc)
            coh_c.append(c_coh if c_coh is not None else np.nan)
        coh_med = float(np.nanmedian(coh_c))
        sc_med = float(np.nanmedian(sc_c)) if not np.all(np.isnan(sc_c)) else float("nan")
        rows.append(dict(n=n, dim=2 ** n,
                         coherent_copies=coh_med,
                         singlecopy_copies=sc_med,
                         ratio=(sc_med / coh_med if coh_med and not np.isnan(sc_med)
                                else float("nan")),
                         seeds=len(seeds),
                         singlecopy_capped_runs=capped,
                         wall_s=round(time.time() - t_n, 1)))
        print(f"  == n={n}: coherent={coh_med:.0f}  single-copy={sc_med}  "
              f"ratio={rows[-1]['ratio']}  capped={capped}/{len(seeds)}  "
              f"[{rows[-1]['wall_s']}s]", flush=True)
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, "learning_separation_scaled.csv")
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("  wrote", p, flush=True)

    # growth-exponent fit for the single-copy arm
    ok = [r for r in rows if not np.isnan(r["singlecopy_copies"]) and r["singlecopy_capped_runs"] == 0]
    if len(ok) >= 3:
        x = np.array([r["n"] for r in ok], dtype=float)
        y = np.log2([r["singlecopy_copies"] for r in ok])
        slope, icpt = np.polyfit(x, y, 1)
        resid = y - (slope * x + icpt)
        print(f"FIT single-copy copies ~ 2^({slope:.3f} n + {icpt:.3f})   "
              f"i.e. base {2**slope:.3f}^n ; max|resid|={np.max(np.abs(resid)):.3f} bits ; "
              f"fitted on n={[int(v) for v in x]}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=list(range(2, 13)))
    ap.add_argument("--m", type=int, default=24)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--mixed-rank", type=int, default=None)
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--max-copies", type=int, default=1 << 16)
    ap.add_argument("--time-cap", type=float, default=180.0,
                    help="seconds per (n,seed) for the single-copy ladder")
    ap.add_argument("--out", default="results/two_copy_separation")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if args.verify:
        verify(); return
    run(args.sizes, args.m, args.seeds, args.out, mixed_rank=args.mixed_rank,
        target=args.target, max_copies=args.max_copies, time_cap=args.time_cap)


if __name__ == "__main__":
    main()
