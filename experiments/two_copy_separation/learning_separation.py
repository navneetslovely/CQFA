
from __future__ import annotations

import argparse
import csv
import os

import numpy as np


# ---------------------------------------------------------------------------
# state ensembles (the two physical classes)
# ---------------------------------------------------------------------------
def haar_pure(n: int, rng) -> np.ndarray:
   
    d = 2 ** n
    v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    v /= np.linalg.norm(v)
    return np.outer(v, v.conj())


def mixed_state(n: int, rng, rank: int | None = None) -> np.ndarray:
    
    d = 2 ** n
    if rank is None:
        return np.eye(d, dtype=complex) / d
    rho = np.zeros((d, d), dtype=complex)
    for _ in range(rank):
        v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        v /= np.linalg.norm(v)
        rho += np.outer(v, v.conj())
    return rho / rank


def purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


def make_dataset(n: int, m: int, rng, mixed_rank=None):
    """Balanced dataset of (rho, purity, label): label 1 = pure, 0 = mixed."""
    data = []
    for _ in range(m // 2):
        r = haar_pure(n, rng); data.append((r, purity(r), 1))
        r = mixed_state(n, rng, rank=mixed_rank); data.append((r, purity(r), 0))
    rng.shuffle(data)
    return data


# ---------------------------------------------------------------------------
# coherent 2-copy reader: purity via destructive SWAP (per-measurement var <= 1)
# ---------------------------------------------------------------------------
def coherent_purity_samples(pur: float, n_copies: int, rng) -> np.ndarray:
   
    p_plus = 0.5 * (1.0 + pur)                          # P(outcome +1)
    return np.where(rng.random(n_copies) < p_plus, 1.0, -1.0)


# ---------------------------------------------------------------------------
# single-copy random-Pauli classical shadows of a (mixed) state
# ---------------------------------------------------------------------------
_PAULI = {
    0: np.array([[0, 1], [1, 0]], dtype=complex),       # X
    1: np.array([[0, -1j], [1j, 0]], dtype=complex),    # Y
    2: np.array([[1, 0], [0, -1]], dtype=complex),      # Z
}
_EIGVEC = {b: np.linalg.eigh(_PAULI[b])[1] for b in (0, 1, 2)}   # columns = |b,0>,|b,1>


def _single_shadow_snapshot(rho: np.ndarray, n: int, rng):
    
    bases = rng.integers(0, 3, size=n)
  
    vecs = [ _EIGVEC[int(bases[q])] for q in range(n) ]
   
    d = 2 ** n
    
    U = np.array([[1.0 + 0j]])
    for q in range(n):
        U = np.kron(U, vecs[q].conj().T)                 # <eigbasis| per qubit
    probs = np.real(np.einsum("ij,jk, ki->i", U, rho, U.conj().T)) if False else None
   
    rho_meas = U @ rho @ U.conj().T
    p = np.real(np.diag(rho_meas))
    p = np.clip(p, 0, None); p /= p.sum()
    idx = int(rng.choice(d, p=p))
    outcomes = np.array([(idx >> (n - 1 - q)) & 1 for q in range(n)], dtype=int)
    return bases, outcomes


def _snapshot_matrix(bases, outcomes, n):
    """Reconstruct the single-copy shadow estimator rho_hat = ⊗_q (3|b,s><b,s| - I)."""
    M = np.array([[1.0 + 0j]])
    for q in range(n):
        v = _EIGVEC[int(bases[q])][:, int(outcomes[q])]
        M = np.kron(M, 3.0 * np.outer(v, v.conj()) - np.eye(2))
    return M


def shadow_purity_estimate(rho: np.ndarray, n: int, n_copies: int, rng) -> float:
    
    snaps = [ _snapshot_matrix(*_single_shadow_snapshot(rho, n, rng), n)
              for _ in range(n_copies) ]
    C = len(snaps)
    if C < 2:
        return float("nan")
    F = np.stack([s.reshape(-1) for s in snaps])            # (C, d^2)
    G = np.stack([s.T.reshape(-1) for s in snaps])          # (C, d^2)
    P = np.real(F @ G.T)                                    # P[i,j] = Tr(S_i S_j)
    tot = float(P.sum() - np.trace(P))                      # off-diagonal pairs only
    return tot / (C * (C - 1))


# ---------------------------------------------------------------------------
# copies-to-accuracy for each learner
# ---------------------------------------------------------------------------
def _accuracy_at_threshold(feats, labels):
    """Best-threshold accuracy of a 1-D feature (the trained classifier)."""
    order = np.argsort(feats)
    fs, ys = np.array(feats)[order], np.array(labels)[order]
    best = 0.5
    for t in (fs[:-1] + fs[1:]) / 2:
        acc = max(np.mean((fs > t) == ys), np.mean((fs <= t) == ys))
        best = max(best, acc)
    return best


def copies_to_accuracy(dataset, learner, budgets, rng, target=0.9):
  
    curve = []
    for C in budgets:
        feats, labels = [], []
        for rho, pur, y in dataset:
            feats.append(learner(rho, pur, C, rng)); labels.append(y)
        acc = _accuracy_at_threshold(feats, labels)
        curve.append((C, acc))
        if acc >= target:
            return C, curve
    return None, curve


def coherent_learner(rho, pur, C, rng):
    return float(np.mean(coherent_purity_samples(pur, C, rng)))


def shadow_learner(rho, pur, C, rng):
    n = int(round(np.log2(rho.shape[0])))
    return shadow_purity_estimate(rho, n, C, rng)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(sizes, m, seeds, out_dir, mixed_rank=None):
    rows = []
    coh_budgets = [2, 4, 8, 16, 32, 64, 128]
    sc_budgets = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    for n in sizes:
        coh_c, sc_c = [], []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            data = make_dataset(n, m, rng, mixed_rank=mixed_rank)
            c_coh, _ = copies_to_accuracy(data, coherent_learner, coh_budgets, rng)
            c_sc, _ = copies_to_accuracy(data, shadow_learner, sc_budgets, rng)
            coh_c.append(c_coh if c_coh is not None else np.nan)
            sc_c.append(c_sc if c_sc is not None else np.nan)
        coh_med = float(np.nanmedian(coh_c))
        sc_med = float(np.nanmedian(sc_c)) if not np.all(np.isnan(sc_c)) else float("nan")
        capped = int(np.sum(np.isnan(sc_c)))              # runs that never reached target
        rows.append(dict(n=n, dim=2 ** n,
                         coherent_copies=coh_med, singlecopy_copies=sc_med,
                         singlecopy_capped_runs=capped,
                         ratio=(sc_med / coh_med if coh_med and not np.isnan(sc_med) else np.nan)))
        print(f"n={n}: coherent 2-copy needs ~{coh_med:.0f} copies (flat); "
              f"single-copy needs ~{sc_med if not np.isnan(sc_med) else '>budget'} "
              f"({capped}/{len(seeds)} runs never reached target within {sc_budgets[-1]})")
    if rows:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, "learning_separation.csv")
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("wrote", p)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--m", type=int, default=24, help="states per dataset (balanced)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--mixed-rank", type=int, default=None, help="None=maximally mixed")
    ap.add_argument("--out", default="results/two_copy_separation")
    args = ap.parse_args(argv)
    run(args.sizes, args.m, args.seeds, args.out, mixed_rank=args.mixed_rank)


if __name__ == "__main__":
    main()
