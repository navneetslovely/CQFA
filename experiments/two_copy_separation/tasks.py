
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix

from .learning_separation import (
    haar_pure, purity, coherent_purity_samples, shadow_purity_estimate,
    _accuracy_at_threshold, copies_to_accuracy, coherent_learner, shadow_learner,
)


# ---------------------------------------------------------------------------
# T2 -- entanglement detection via subsystem purity Tr(rho_A^2)
# ---------------------------------------------------------------------------
def _haar_ket(n, rng):
    v = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
    return v / np.linalg.norm(v)


def make_entanglement_dataset(nA, nB, m, rng):

    data = []
    for _ in range(m // 2):
        # product |a>|b> -> rho_A pure
        a = _haar_ket(nA, rng)
        rhoA = np.outer(a, a.conj())
        data.append((rhoA, purity(rhoA), 1))
        # entangled: Haar on A+B, trace out B -> mixed rho_A
        psi = _haar_ket(nA + nB, rng)
        rhoA = partial_trace(DensityMatrix(psi), list(range(nA, nA + nB))).data
        data.append((rhoA, purity(rhoA), 0))
    rng.shuffle(data)
    return data


def run_entanglement(sizes, m, seeds, out_dir):
    """sizes = list of nA (subsystem size); nB = nA. Same copies-to-accuracy compare."""
    rows = []
    coh_b = [2, 4, 8, 16, 32, 64, 128]
    sc_b = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    for nA in sizes:
        coh, sc = [], []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            data = make_entanglement_dataset(nA, nA, m, rng)
            c, _ = copies_to_accuracy(data, coherent_learner, coh_b, rng)
            s, _ = copies_to_accuracy(data, shadow_learner, sc_b, rng)
            coh.append(c if c else np.nan); sc.append(s if s else np.nan)
        cm = float(np.nanmedian(coh)); sm = float(np.nanmedian(sc)) if not np.all(np.isnan(sc)) else np.nan
        rows.append(dict(task="entanglement", nA=nA, dimA=2 ** nA,
                         coherent_copies=cm, singlecopy_copies=sm,
                         singlecopy_capped=int(np.sum(np.isnan(sc))),
                         ratio=(sm / cm if cm and not np.isnan(sm) else np.nan)))
        print(f"[entanglement] nA={nA}: coherent 2-copy ~{cm:.0f} (flat); "
              f"single-copy ~{sm if not np.isnan(sm) else '>budget'}")
    _write(out_dir, "entanglement.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# T3 -- state overlap / identity testing via Tr(rho sigma) (SWAP test)
# ---------------------------------------------------------------------------
def coherent_overlap_samples(ov: float, n_copies: int, rng):
    """Destructive-SWAP outcomes on rho (x) sigma: +/-1 with mean Tr(rho sigma)=ov."""
    p_plus = 0.5 * (1.0 + ov)
    return np.where(rng.random(n_copies) < p_plus, 1.0, -1.0)


def make_overlap_dataset(n, m, rng):

    data = []
    for _ in range(m // 2):
        a = _haar_ket(n, rng)
        ra = np.outer(a, a.conj())
        data.append(((ra, ra), 1.0, 1))                   # equal -> overlap 1
        b = _haar_ket(n, rng)
        rb = np.outer(b, b.conj())
        ov = float(np.real(np.trace(ra @ rb)))
        data.append(((ra, rb), ov, 0))                    # independent -> overlap ~2^-n
    rng.shuffle(data)
    return data


def _shadow_overlap_estimate(rho, sigma, n, n_copies, rng):

    from .learning_separation import _snapshot_matrix, _single_shadow_snapshot
    half = max(1, n_copies // 2)
    S = [_snapshot_matrix(*_single_shadow_snapshot(rho, n, rng), n) for _ in range(half)]
    T = [_snapshot_matrix(*_single_shadow_snapshot(sigma, n, rng), n) for _ in range(half)]
    F = np.stack([s.reshape(-1) for s in S])
    G = np.stack([t.T.reshape(-1) for t in T])
    return float(np.real(F @ G.T).mean())


def run_overlap(sizes, m, seeds, out_dir):
    rows = []
    coh_b = [2, 4, 8, 16, 32, 64, 128]
    sc_b = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    for n in sizes:
        coh, sc = [], []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            data = make_overlap_dataset(n, m, rng)
            c, _ = copies_to_accuracy(
                data, lambda pair, ov, C, r: float(np.mean(coherent_overlap_samples(ov, C, r))),
                coh_b, rng)
            s, _ = copies_to_accuracy(
                data, lambda pair, ov, C, r: _shadow_overlap_estimate(pair[0], pair[1], n, C, r),
                sc_b, rng)
            coh.append(c if c else np.nan); sc.append(s if s else np.nan)
        cm = float(np.nanmedian(coh)); sm = float(np.nanmedian(sc)) if not np.all(np.isnan(sc)) else np.nan
        rows.append(dict(task="overlap", n=n, dim=2 ** n,
                         coherent_copies=cm, singlecopy_copies=sm,
                         singlecopy_capped=int(np.sum(np.isnan(sc))),
                         ratio=(sm / cm if cm and not np.isnan(sm) else np.nan)))
        print(f"[overlap] n={n}: coherent 2-copy ~{cm:.0f} (flat); "
              f"single-copy ~{sm if not np.isnan(sm) else '>budget'}")
    _write(out_dir, "overlap.csv", rows)
    return rows


def _write(out_dir, name, rows):
    if not rows:
        return
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", p)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["entanglement", "overlap", "both"], default="both")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--m", type=int, default=24)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/two_copy_separation")
    args = ap.parse_args(argv)
    if args.task in ("entanglement", "both"):
        run_entanglement(args.sizes, args.m, args.seeds, args.out)
    if args.task in ("overlap", "both"):
        run_overlap(args.sizes, args.m, args.seeds, args.out)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# X7 -- resilience of the 2-copy purity separation under depolarizing noise
# ---------------------------------------------------------------------------
def run_purity_resilience(sizes, m, seeds, p_deps, out_dir):

    from .learning_separation import (make_dataset, coherent_learner, shadow_learner,
                                       copies_to_accuracy, purity)
    rows = []
    coh_b = [2, 4, 8, 16, 32, 64, 128]
    sc_b = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    for n in sizes:
        d = 2 ** n
        for p in p_deps:
            coh, sc = [], []
            for sd in seeds:
                rng = np.random.default_rng(sd)
                data = make_dataset(n, m, rng)
                noisy = []
                for rho, pur, y in data:
                    rn = (1 - p) * rho + p * np.eye(d) / d      # depolarized copy
                    noisy.append((rn, purity(rn), y))
                c, _ = copies_to_accuracy(noisy, coherent_learner, coh_b, rng)
                s, _ = copies_to_accuracy(noisy, shadow_learner, sc_b, rng)
                coh.append(c if c else np.nan); sc.append(s if s else np.nan)
            cm = float(np.nanmedian(coh)); sm = float(np.nanmedian(sc)) if not np.all(np.isnan(sc)) else np.nan
            rows.append(dict(n=n, p_depol=p, coherent_copies=cm, singlecopy_copies=sm,
                             ratio=(sm / cm if cm and not np.isnan(sm) else np.nan)))
            print(f"[purity-resilience] n={n} depol={p:.3f}: coherent ~{cm:.0f}  single-copy ~{sm}")
    _write(out_dir, "purity_resilience.csv", rows)
    return rows
