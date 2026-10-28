"""
U1 random-basis arm -- the untrained control arm.

WHY THIS FILE EXISTS
--------------------
This arm is a condition on the abstract-level claim. A shallow trainable circuit
readout recovers ~99% of the gap that the fixed Fourier readout leaves. But a
single DATA-INDEPENDENT Haar-random basis, with zero fitting shots and no training
whatsoever, already recovers most of the same gap at exact readout. The trainable
readout must therefore be reported against that arm, not against the fixed Fourier
basis alone.

This arm measures ~0.956 for that arm, but only in scratch scripts. Integrity
rule 1 forbids citing a number that is not in a committed CSV, so this module
measures it properly and writes it down.

WHY THE RANDOM ARM WORKS AT ALL, AND WHY IT IS NOT A REFUTATION
---------------------------------------------------------------
It is explained by the same rank fact the abstract states. The fixed generalized
Fourier basis is rank-deficient on non-abelian G: it reads only sum_lambda d_lambda
of the |G| real parameters. A Haar-random basis is full-rank almost surely, so it
sees the missing directions -- it is simply badly CONDITIONED rather than blind.
That is why it does well at exact readout and, collapses at
finite shots. Both arms are measured here at both budgets so the paper can state the
trade : Fourier is well-conditioned but rank-deficient, random is full-rank
but ill-conditioned, and only the learned readout is both.

PRE-REGISTERED, TWO-SIDED (fixed before first run):
  The abstract's claim SURVIVES if the learned circuit still beats a random basis at
    exact readout by mean(delta) > 2*se over >= 5 seeds on non-abelian groups.
  It is KILLED if a random basis matches or beats the learned circuit at exact
    readout -- in which case "a trainable circuit readout recovers the gap" is not
    attributable to training at all, and the abstract sentence must be withdrawn
    before submission, not after.

Reads : results/readout_basis/u1_shotmatched_percell.csv   (committed, unchanged)
Writes: results/readout_basis/u1_random_basis_arm.csv      (NEW; nothing overwritten)
"""
from __future__ import annotations

import csv
import math
import os

import numpy as np
from scipy import stats as sps

import experiments.readout_basis.u1_shotmatched_nonabelian as U

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(REPO, "results", "readout_basis")
PERCELL = os.path.join(OUTDIR, "u1_shotmatched_percell.csv")
OUT = os.path.join(OUTDIR, "u1_random_basis_arm.csv")

NONAB = ["S3", "D4", "Q8", "D5", "A4", "D6", "D10", "F21", "S4"]
ABEL = ["C6", "C8", "C12"]
TASK = "generic_W"
SEEDS = [0, 1, 2, 3, 4]
N_DRAWS = 8          # independent Haar draws per (group, seed); fixed before running


def haar(D, rng):
    z = (rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))) / math.sqrt(2.0)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def main():
    percell = {}
    for r in csv.DictReader(open(PERCELL)):
        percell.setdefault((r["group"], r["task"], r["shots"]), []).append(r)

    def arr(key, col):
        sub = sorted(percell[key], key=lambda r: int(r["seed"]))
        return np.array([float(r[col]) for r in sub])

    rows = []
    for S, slab in ((256, "256"), (None, "exact")):
        for name in NONAB + ABEL:
            T, m = U.group_table(name)
            L, Rop = U.regular_ops(T)
            n = max(1, int(math.ceil(math.log2(m))))
            D = 1 << n
            iu = U._pair_idx(m)
            F_m = U.fourier_basis(L, Rop, U.CANONICAL_BASIS)

            rand_acc = np.zeros(len(SEEDS))
            for si, seed in enumerate(SEEDS):
                rhos, y, _ = U.make_dataset(m, L, F_m, TASK, seed)
                Psi = U.phi(rhos, iu)
                Ptr, Pte = Psi[:U.N_TRAIN], Psi[U.N_TRAIN:]
                ytr, yte = y[:U.N_TRAIN], y[U.N_TRAIN:]
                g = np.random.default_rng(770077 + 31 * seed + 7 * D)
                # mean over independent draws: the arm is a SINGLE random basis,
                # so we report the expected single-draw result, not the best of
                # several (which would be another search).
                accs = []
                for _ in range(N_DRAWS):
                    Rr = haar(D, g)
                    _, te, _ = U.fit_logreg(
                        U.sample_shots(U.probs(Ptr, Rr, m, iu), S, g), ytr,
                        U.sample_shots(U.probs(Pte, Rr, m, iu), S, g), yte)
                    accs.append(te)
                rand_acc[si] = float(np.mean(accs))

            circ = arr((name, TASK, slab), "acc_test_learned_circuit")
            mf = arr((name, TASK, slab), "acc_test_modeF_fixed")
            bb = arr((name, TASK, slab), "acc_test_modeF_bestbasis")
            uni = arr((name, TASK, slab), "acc_test_learned_unitary")

            def paired(a, b):
                d = a - b
                return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))

            d_cr = paired(circ, rand_acc)
            rows.append(dict(
                group=name, abelian=int(name in ABEL), order=m, shots=slab,
                n_seeds=len(SEEDS), n_haar_draws=N_DRAWS,
                mean_acc_random_basis=float(rand_acc.mean()),
                se_acc_random_basis=float(rand_acc.std(ddof=1) / math.sqrt(len(rand_acc))),
                mean_acc_modeF_fixed=float(mf.mean()),
                mean_acc_modeF_bestbasis=float(bb.mean()),
                mean_acc_learned_circuit=float(circ.mean()),
                mean_acc_learned_unitary=float(uni.mean()),
                mean_delta_circ_minus_random=d_cr[0], se_delta_circ_minus_random=d_cr[1],
                circ_beats_random_2se=int(d_cr[0] > 2 * d_cr[1]),
                random_minus_modeF_fixed=float((rand_acc - mf).mean()),
                # what fraction of the fixed-Fourier-to-tomographic gap the random
                # basis closes, using learned_unitary as the tomographic ceiling
                frac_of_gap_closed_by_random=(
                    float((rand_acc.mean() - mf.mean()) / (uni.mean() - mf.mean()))
                    if abs(uni.mean() - mf.mean()) > 1e-9 else float("nan")),
                frac_of_gap_closed_by_circuit=(
                    float((circ.mean() - mf.mean()) / (uni.mean() - mf.mean()))
                    if abs(uni.mean() - mf.mean()) > 1e-9 else float("nan")),
            ))
            print(f"  [{slab:>5}] {name:>4} random={rand_acc.mean():.4f} "
                  f"modeF={mf.mean():.4f} circ={circ.mean():.4f} uni={uni.mean():.4f} "
                  f"circ-rand={d_cr[0]:+.4f}+-{d_cr[1]:.4f}", flush=True)

    assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n=== VERDICT (non-abelian, learned circuit vs a single random basis) ===")
    for slab in ("exact", "256"):
        sub = [r for r in rows if r["shots"] == slab and not r["abelian"]]
        npass = sum(r["circ_beats_random_2se"] for r in sub)
        pooled = np.array([r["mean_delta_circ_minus_random"] for r in sub])
        t, p = sps.ttest_1samp(pooled, 0, alternative="greater")
        rnd = np.mean([r["mean_acc_random_basis"] for r in sub])
        frac = np.mean([r["frac_of_gap_closed_by_random"] for r in sub])
        print(f"  S={slab:>5}: random basis mean acc {rnd:.4f} "
              f"(closes {frac*100:.0f}% of the fixed->tomographic gap); "
              f"circuit beats it {npass}/9, pooled {pooled.mean():+.4f}, p={p:.2e}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
