"""Strong versus weak Fourier sampling as frame readouts.

The fixed Mode-F readout measures the frame register in the generalized Fourier basis and
records the irrep label: this is weak Fourier sampling. A strong-Fourier-sampling readout
also measures a basis inside each irrep block. Both are a frame unitary followed by a
computational-basis measurement, so both lie in the readout family of the paper, and this
module asks how the trained readouts relate to them. Every arm sees the same data, seeds,
head and shot budgets as the shot-matched sweep.

ARMS
----
  weak            R = F_G                      Mode F fixed; reads irrep labels only
  strong_dft      R = F_G B,  B = DFT inside every irrep block of F_G^dag rho F_G
  strong_random   R = F_G B,  B Haar-random inside every block; mean over N_DRAWS draws
                  (strong Fourier sampling in a random basis)
  strong_learned  R = F_G B,  B a block-diagonal unitary trained end to end with the head,
                  warm-started at B = I on one restart and random on the other, with the
                  optimiser, epochs and shot accounting of the learned-unitary arm; the
                  learned readout CONSTRAINED to strong Fourier sampling
  random_full     R Haar-random on the whole frame register (u1_random_basis_arm.csv);
                  this mixes irrep sectors and is NOT strong Fourier sampling
  learned_unitary, learned_circuit               (u1_shotmatched_percell.csv)

For abelian G every irrep block is 1x1, B is a diagonal phase, and every strong arm equals
weak exactly; this is asserted and recorded (the built-in control).

Fixed and strong_learned arms are read out by a fresh logistic head on the measured
outcome frequencies, exactly as the random-basis arm is; the learned_unitary and
learned_circuit values are the sweep's own.

ACCESSIBLE FEATURE RANK
-----------------------
Every arm feeds a linear head on outcome frequencies, so what a readout can use is the span
of its outcome projectors restricted to the |G|-dimensional space of frame states. That span
has a computable dimension, `feature_rank`, and it is bounded by the irrep structure:
  weak    : sum_lambda d_lambda                       (block diagonals)
  strong  : sum_lambda (d_lambda^2 - d_lambda + 1)    (any sector-preserving bases, one per copy:
            each orthonormal basis of a d x d block adds at most d-1 directions beyond I)
  mixing  : |G| = sum_lambda d_lambda^2               (sector-mixing readouts, when D >= |G|)
so strong Fourier sampling falls short of the full frame state by sum_lambda (d_lambda - 1)
real parameters on every non-abelian group, and a sector-mixing trainable readout is what
closes that remainder. The ranks are measured per arm and reported with the bounds.

INCLUSION CHECK
---------------
Every strong-sampling basis R = F_G B is a unitary, so it is reproduced exactly by the
learned-unitary parameterisation R = expm(iH): the round trip through theta_from_herm and
unitary_from_theta is reported as `inclusion_error`. The unconstrained trained head
therefore contains strong Fourier sampling as a special case. F_G is not in the shallow
circuit family, so the circuit's relation to strong sampling is empirical and is read off
the paired deltas.

SHOT-MATCHED COMPARISON
-----------------------
The submitted paper reports that at 256 shots a best-of-128 pool of fixed Fourier bases,
selected on training accuracy, ties the learned circuit, and claims no shot-matched
advantage. Two versions of that pool are built here per seed, with the same 128 bases and
random streams as u1_abstract_evidence:

  bestof128_train  the basis with the highest TRAINING accuracy of its fitted head
                   (the submitted paper's comparator; its pooled mean must reproduce the
                   committed u1_pool_sweep.csv)
  bestof128_val    the basis with the highest accuracy on a held-out VALIDATION split of
                   the training states (N_FIT = 300 fit, N_VAL = 100 validate), refit on
                   the full training set before the test evaluation

Selecting on the training accuracy of a fitted head among equivalent bases picks the shot
realisation on which the head overfits most, so on an abelian group, where all 128 bases
give the same outcome distribution, bestof128_train sits BELOW a single fixed basis on the
test set. The validation-selected pool removes that bias and is the comparator the verdict
is read against; the train-selected pool is reported alongside for continuity.

SEEDS
-----
The sweep's arms (learned_unitary, learned_circuit, bestof8, random_full) exist for five
seeds. Every arm this module computes (weak, the three strong arms, both pools) is run for
N_SEEDS_MODULE = 20 seeds; comparisons with sweep arms are paired on the first five. The
reason is the control test. With five seeds, two statistically identical single-basis
evaluations on an abelian group (weak and bestof128_val) were observed to differ by more
than two estimated standard errors in one group of three, because a standard error
estimated from five paired differences is itself uncertain; the 2 se control gate then has
a false-positive rate of order 15-20 percent and cannot distinguish bias from noise. Twenty
seeds make the gate informative. Both the five-seed and the twenty-seed verdicts are
written to the summary.

Criterion, fixed before the first run with the validation pool and mirroring the sweep's
track rule: the block-structured readout has a shot-matched advantage over fixed-basis
search iff at 256 shots mean(delta) > 2 se against bestof128_val in at least 5 of the 9
non-abelian groups, and no abelian control differs from bestof128_val by more than 2 se in
either direction. It is read at twenty seeds; the five-seed reading is reported alongside.

Writes results/readout_basis/u4_strong_fourier_sampling.csv (per group x shots) and
results/readout_basis/u4_strong_fourier_sampling_summary.csv (pooled over groups).

Run::

    python -m experiments.readout_basis.u4_strong_fourier_sampling
"""
from __future__ import annotations

import csv
import math
import os

import numpy as np
from scipy.linalg import logm

import experiments.readout_basis.u1_shotmatched_nonabelian as U
from experiments.coherent_surplus.surplus_span_ranks import isotypic_structure
from cqfa.paths import RESULTS, ensure

OUTDIR = str(ensure(RESULTS / "readout_basis"))
PERCELL = os.path.join(OUTDIR, "u1_shotmatched_percell.csv")
RANDOM_ARM = os.path.join(OUTDIR, "u1_random_basis_arm.csv")
OUT = os.path.join(OUTDIR, "u4_strong_fourier_sampling.csv")
OUT_SUMMARY = os.path.join(OUTDIR, "u4_strong_fourier_sampling_summary.csv")

NONAB = U.NONABELIAN
ABEL = U.ABELIAN
TASK = "generic_W"
SEEDS = U.SEEDS                  # the sweep's five seeds; sweep arms exist only for these
N_SEEDS_MODULE = 20
SEEDS_MODULE = list(range(N_SEEDS_MODULE))
N_DRAWS = 8                      # Haar draws per (group, seed), as in the random-basis arm
BLOCK_TOL = 1e-9
N_RHO_FOR_BLOCKS = 32
POOL_K = 128                     # the fixed-basis pool the submitted paper compares against
POOL_RNG = 20250722              # identical stream to u1_abstract_evidence
N_FIT = 300                      # validation-selected pool: head fit on 300 training states
N_VAL = 100                      #                            selected on the other 100


# ---------------------------------------------------------------------------
# irrep blocks of the Fourier-transformed frame state
# ---------------------------------------------------------------------------
def fourier_blocks(F_m, rhos, tol=BLOCK_TOL):
    """Index sets of the irrep blocks of F^dag rho F, as connected components of the
    |entry| > tol graph accumulated over several frame states."""
    m = F_m.shape[0]
    A = np.zeros((m, m), dtype=bool)
    for rho in rhos:
        A |= np.abs(F_m.conj().T @ rho @ F_m) > tol
    A |= A.T
    seen, comps = set(), []
    for i in range(m):
        if i in seen:
            continue
        stack, comp = [i], []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            stack.extend(int(v) for v in np.where(A[u])[0] if v not in seen)
        comps.append(sorted(comp))
    return sorted(comps, key=lambda c: (len(c), c[0]))


def dft(k):
    a = np.arange(k)
    return np.exp(-2j * np.pi * np.outer(a, a) / k) / math.sqrt(k)


def haar(k, rng):
    z = (rng.normal(size=(k, k)) + 1j * rng.normal(size=(k, k))) / math.sqrt(2.0)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def block_unitary(D, blocks, mats):
    """D x D unitary equal to mats[b] on block b and the identity elsewhere."""
    B = np.eye(D, dtype=complex)
    for idx, Mb in zip(blocks, mats):
        B[np.ix_(idx, idx)] = Mb
    return B


# ---------------------------------------------------------------------------
# readout evaluation with a fresh head (the random-basis arm's convention)
# ---------------------------------------------------------------------------
def head_accuracy(R, Ptr, ytr, Pte, yte, m, iu, S, rng):
    _, te, _ = U.fit_logreg(U.sample_shots(U.probs(Ptr, R, m, iu), S, rng), ytr,
                            U.sample_shots(U.probs(Pte, R, m, iu), S, rng), yte)
    return te


# ---------------------------------------------------------------------------
# strong_learned: block-constrained unitary trained end to end
# ---------------------------------------------------------------------------
def train_block_readout(Psi_tr, ytr, m, D, blocks, F_embed, S, seed, iu):
    """Adam on (block parameters, w, b) with the straight-through shot estimator, the
    schedule and restart rule of U.train_readout; R = F_embed @ B(theta), B block-diagonal."""
    rng = np.random.default_rng(900_000 + 131 * seed + 7 * D + 11)
    Ntr = Psi_tr.shape[0]
    yv = ytr.astype(float)
    sizes = [len(b) for b in blocks]
    npb = [k * k for k in sizes]                       # real parameters per block (Hermitian k x k)
    npR = sum(npb)
    offs = np.cumsum([0] + npb)

    def build(thR):
        mats = [U.unitary_from_theta(thR[offs[i]:offs[i + 1]], k) for i, k in enumerate(sizes)]
        return F_embed @ block_unitary(D, blocks, mats)

    best = None
    for restart in range(U.N_RESTARTS_R):
        warm = (restart == 0)
        thR = np.zeros(npR) if warm else 0.6 * rng.normal(size=npR)
        lr = U.LR_WARM if warm else U.LR_RAND
        th = np.concatenate([thR, np.zeros(D), [0.0]])
        mt, vt = np.zeros_like(th), np.zeros_like(th)
        for ep in range(U.EPOCHS):
            R = build(th[:npR])
            p = U.probs(Psi_tr, R, m, iu)
            f = U.sample_shots(p, S, rng)
            mu, sd = f.mean(0), np.where(f.std(0) < 1e-12, 1.0, f.std(0))
            X = (f - mu) / sd
            wv, bv = th[npR:npR + D], th[npR + D]
            z = X @ wv + bv
            s = 1.0 / (1.0 + np.exp(-z))
            gz = (s - yv) / Ntr
            gw = X.T @ gz + 2 * U.L2 * wv
            gb = float(gz.sum())
            Gp = np.outer(gz, wv) / sd
            Z = U.dprobs_to_Z(Gp, Psi_tr, R, m, iu)       # dL/d conj(R)
            ZB = F_embed.conj().T @ Z                      # dL/d conj(B), since R = F B
            gR = np.concatenate([
                U.grad_theta_from_dR(ZB[np.ix_(idx, idx)], th[offs[i]:offs[i + 1]], len(idx))
                for i, idx in enumerate(blocks)])
            g = np.concatenate([gR, gw, [gb]])
            lr_t = lr * 0.5 * (1.0 + math.cos(math.pi * ep / U.EPOCHS))
            mt = 0.9 * mt + 0.1 * g
            vt = 0.999 * vt + 0.001 * g * g
            mh = mt / (1 - 0.9 ** (ep + 1))
            vh = vt / (1 - 0.999 ** (ep + 1))
            th = th - lr_t * mh / (np.sqrt(vh) + 1e-8)
        R = build(th[:npR])
        f = U.sample_shots(U.probs(Psi_tr, R, m, iu), S, rng)
        mu, sd = f.mean(0), np.where(f.std(0) < 1e-12, 1.0, f.std(0))
        z = ((f - mu) / sd) @ th[npR:npR + D] + th[npR + D]
        loss = float(np.mean(np.logaddexp(0.0, z) - yv * z))
        if best is None or loss < best[0]:
            best = (loss, R)
    return best[1]


def feature_rank(R, Psi, m, iu, tol=1e-9):
    """Dimension of the span of the outcome functionals rho -> (R^dag rho R)_jj over the frame
    states in Psi (which span the |G|-dimensional circulant space for generic data)."""
    P = U.probs(Psi, R, m, iu)
    P = P - P.mean(axis=0, keepdims=True)           # affine head: constants are free
    sv = np.linalg.svd(P, compute_uv=False)
    return int((sv > tol * sv[0]).sum()) + 1 if sv[0] > 0 else 1


def pool_best_of_k(name, S, K=POOL_K):
    """Per-seed test accuracy of the best-of-K fixed GFT basis under two selection rules.

    Returns (train_selected, val_selected), each an array over seeds. The train-selected rule
    and its random streams are identical to u1_abstract_evidence. The validation rule fits the
    head on the first N_FIT training states, scores each basis on the remaining N_VAL, picks
    the best, then refits the head on all training states for the test evaluation; the extra
    refit is one measurement round for the selected basis only.
    """
    T, m = U.group_table(name)
    L, Rop = U.regular_ops(T)
    n = max(1, int(math.ceil(math.log2(m))))
    D = 1 << n
    iu = U._pair_idx(m)
    F0 = U.fourier_basis(L, Rop, 0)
    out_tr = np.zeros(len(SEEDS_MODULE))
    out_val = np.zeros(len(SEEDS_MODULE))
    for si, seed in enumerate(SEEDS_MODULE):
        rhos, y, _ = U.make_dataset(m, L, F0, TASK, seed)
        Psi = U.phi(rhos, iu)
        Ptr, Pte = Psi[:U.N_TRAIN], Psi[U.N_TRAIN:]
        ytr, yte = y[:U.N_TRAIN], y[U.N_TRAIN:]
        Pfit, Pval = Ptr[:N_FIT], Ptr[N_FIT:N_FIT + N_VAL]
        yfit, yval = ytr[:N_FIT], ytr[N_FIT:N_FIT + N_VAL]
        g = np.random.default_rng(POOL_RNG + 911 * seed + 13 * D)
        gv = np.random.default_rng(POOL_RNG + 911 * seed + 13 * D + 1)
        tr = np.zeros(K); te = np.zeros(K); val = np.zeros(K)
        Fes = []
        for b in range(K):
            Fb = U.fourier_basis(L, Rop, b)
            Fe = np.eye(D, dtype=complex)
            Fe[:m, :m] = Fb
            Fes.append(Fe)
            tr[b], te[b], _ = U.fit_logreg(U.sample_shots(U.probs(Ptr, Fe, m, iu), S, g), ytr,
                                           U.sample_shots(U.probs(Pte, Fe, m, iu), S, g), yte)
            _, val[b], _ = U.fit_logreg(U.sample_shots(U.probs(Pfit, Fe, m, iu), S, gv), yfit,
                                        U.sample_shots(U.probs(Pval, Fe, m, iu), S, gv), yval)
        out_tr[si] = te[int(np.argmax(tr))]
        Fsel = Fes[int(np.argmax(val))]
        _, out_val[si], _ = U.fit_logreg(U.sample_shots(U.probs(Ptr, Fsel, m, iu), S, gv), ytr,
                                         U.sample_shots(U.probs(Pte, Fsel, m, iu), S, gv), yte)
    return out_tr, out_val


def inclusion_error(R, D):
    """|| expm(i H) - R || for H the Hermitian log of R, through the sweep's parameterisation."""
    H = -1j * logm(R)
    H = (H + H.conj().T) / 2.0
    R2 = U.unitary_from_theta(U.theta_from_herm(H, D), D)
    return float(np.linalg.norm(R2 - R))


# ---------------------------------------------------------------------------
def main():
    percell = {}
    for r in csv.DictReader(open(PERCELL)):
        percell.setdefault((r["group"], r["task"], r["shots"]), []).append(r)
    randarm = {(r["group"], r["shots"]): r for r in csv.DictReader(open(RANDOM_ARM))}

    def arr(key, col):
        sub = sorted(percell[key], key=lambda r: int(r["seed"]))
        return np.array([float(r[col]) for r in sub])

    def paired(a, b):
        d = np.asarray(a) - np.asarray(b)
        return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))

    rows = []
    for S, slab in ((None, "exact"), (256, "256")):
        for name in NONAB + ABEL:
            T, m = U.group_table(name)
            L, Rop = U.regular_ops(T)
            n = max(1, int(math.ceil(math.log2(m))))
            D = 1 << n
            iu = U._pair_idx(m)
            F_m = U.fourier_basis(L, Rop, U.CANONICAL_BASIS)
            F_embed = np.eye(D, dtype=complex)
            F_embed[:m, :m] = F_m
            abelian = name in ABEL

            acc = {k: np.zeros(len(SEEDS_MODULE)) for k in
                   ("weak", "strong_dft", "strong_random", "strong_learned")}
            blocks = None
            incl_err = 0.0
            strong_eq_weak = True
            for si, seed in enumerate(SEEDS_MODULE):
                rhos, y, _ = U.make_dataset(m, L, F_m, TASK, seed)
                if blocks is None:
                    blocks = fourier_blocks(F_m, rhos[:N_RHO_FOR_BLOCKS])
                    sizes = [len(b) for b in blocks]
                    assert sum(sizes) == m, (name, sizes)
                    irr = isotypic_structure(list(L))            # left-regular: m_lambda = d_lambda
                    assert irr is not None, name
                    dims = [dl for dl, _ in irr]
                    expected = sorted(d for d in dims for _ in range(d))
                    assert sorted(sizes) == expected, (name, sorted(sizes), expected)
                    bound_weak = sum(dims)
                    bound_strong = sum(d * d - d + 1 for d in dims)
                    bound_full = sum(d * d for d in dims)
                    assert bound_full == m
                Psi = U.phi(rhos, iu)
                Ptr, Pte = Psi[:U.N_TRAIN], Psi[U.N_TRAIN:]
                ytr, yte = y[:U.N_TRAIN], y[U.N_TRAIN:]
                g = np.random.default_rng(770_077 + 31 * seed + 7 * D + 3)

                acc["weak"][si] = head_accuracy(F_embed, Ptr, ytr, Pte, yte, m, iu, S, g)
                if si == 0:
                    ranks = dict(weak=feature_rank(F_embed, Psi, m, iu))

                R_dft = F_embed @ block_unitary(D, blocks, [dft(len(b)) for b in blocks])
                acc["strong_dft"][si] = head_accuracy(R_dft, Ptr, ytr, Pte, yte, m, iu, S, g)
                incl_err = max(incl_err, inclusion_error(R_dft, D))
                if si == 0:
                    ranks["strong_dft"] = feature_rank(R_dft, Psi, m, iu)
                    ranks["random_full"] = feature_rank(
                        F_embed @ haar(D, np.random.default_rng(5 + D)), Psi, m, iu)

                draws = []
                for _ in range(N_DRAWS):
                    R_r = F_embed @ block_unitary(D, blocks, [haar(len(b), g) for b in blocks])
                    draws.append(head_accuracy(R_r, Ptr, ytr, Pte, yte, m, iu, S, g))
                    if si == 0 and len(draws) == 1:
                        ranks["strong_random"] = feature_rank(R_r, Psi, m, iu)
                    if abelian:
                        pw = U.probs(Pte, F_embed, m, iu)
                        pr = U.probs(Pte, R_r, m, iu)
                        strong_eq_weak &= bool(np.max(np.abs(pw - pr)) < 1e-10)
                acc["strong_random"][si] = float(np.mean(draws))

                R_l = train_block_readout(Ptr, ytr, m, D, blocks, F_embed, S, seed, iu)
                acc["strong_learned"][si] = head_accuracy(R_l, Ptr, ytr, Pte, yte, m, iu, S, g)
                if si == 0:
                    ranks["strong_learned"] = feature_rank(R_l, Psi, m, iu)

            key = (name, TASK, slab)
            circ = arr(key, "acc_test_learned_circuit")
            uni = arr(key, "acc_test_learned_unitary")
            mf_sweep = arr(key, "acc_test_modeF_fixed")
            bb8 = arr(key, "acc_test_modeF_bestbasis")
            rnd = float(randarm[(name, slab)]["mean_acc_random_basis"])
            pool, pool_val = pool_best_of_k(name, S)

            k5 = len(SEEDS)                         # sweep arms exist for these seeds only
            row = dict(group=name, abelian=int(abelian), order=m, shots=slab,
                       n_seeds_module=len(SEEDS_MODULE), n_seeds_sweep=k5, n_haar_draws=N_DRAWS,
                       n_irrep_blocks=len(blocks),
                       block_sizes=";".join(str(len(b)) for b in blocks),
                       mean_acc_weak=float(acc["weak"].mean()),
                       mean_acc_weak_first5=float(acc["weak"][:k5].mean()),
                       mean_acc_weak_sweep=float(mf_sweep.mean()),
                       mean_acc_strong_dft=float(acc["strong_dft"].mean()),
                       se_acc_strong_dft=float(acc["strong_dft"].std(ddof=1) / math.sqrt(len(SEEDS_MODULE))),
                       mean_acc_strong_random=float(acc["strong_random"].mean()),
                       se_acc_strong_random=float(acc["strong_random"].std(ddof=1) / math.sqrt(len(SEEDS_MODULE))),
                       mean_acc_strong_learned=float(acc["strong_learned"].mean()),
                       se_acc_strong_learned=float(acc["strong_learned"].std(ddof=1) / math.sqrt(len(SEEDS_MODULE))),
                       mean_acc_strong_learned_first5=float(acc["strong_learned"][:k5].mean()),
                       mean_acc_random_full=rnd,
                       mean_acc_learned_unitary=float(uni.mean()),
                       mean_acc_learned_circuit=float(circ.mean()),
                       mean_acc_bestof8_fixed=float(bb8.mean()),
                       mean_acc_bestof128_fixed=float(pool.mean()),
                       mean_acc_bestof128_fixed_first5=float(pool[:k5].mean()),   # reproduces u1_pool_sweep.csv
                       mean_acc_bestof128_val=float(pool_val.mean()),
                       mean_acc_bestof128_val_first5=float(pool_val[:k5].mean()),
                       params_block_readout=sum(len(b) ** 2 for b in blocks),
                       params_circuit=8 * n, params_unitary=D * D,
                       inclusion_error=incl_err,
                       strong_equals_weak_abelian=(int(strong_eq_weak) if abelian else ""),
                       frame_qubits=n, D=D,
                       bound_rank_weak=bound_weak, bound_rank_strong=bound_strong,
                       bound_rank_full=bound_full,
                       rank_weak=ranks["weak"], rank_strong_dft=ranks["strong_dft"],
                       rank_strong_random=ranks["strong_random"],
                       rank_strong_learned=ranks["strong_learned"],
                       rank_random_full=ranks["random_full"])
            SL, W, SD, SR = acc["strong_learned"], acc["weak"], acc["strong_dft"], acc["strong_random"]
            for a, b, lab in ((SD, W, "strong_dft_minus_weak"),
                              (SR, W, "strong_random_minus_weak"),
                              (SL, W, "strong_learned_minus_weak"),
                              (SL, SD, "strong_learned_minus_strong_dft"),
                              (SL, pool, "strong_learned_minus_bestof128"),
                              (SL, pool_val, "strong_learned_minus_bestof128_val"),
                              (W, pool_val, "weak_minus_bestof128_val"),
                              (W, pool, "weak_minus_bestof128")):
                mu, se = paired(a, b)                                   # 20 seeds
                row[f"delta_{lab}"] = mu
                row[f"se_{lab}"] = se
                mu5, se5 = paired(a[:k5], b[:k5])                      # first 5, for the pre-registered reading
                row[f"delta5_{lab}"] = mu5
                row[f"se5_{lab}"] = se5
            for a, b, lab in ((circ, SL[:k5], "circuit_minus_strong_learned"),
                              (uni, SL[:k5], "unitary_minus_strong_learned"),
                              (circ, SD[:k5], "circuit_minus_strong_dft"),
                              (SL[:k5], bb8, "strong_learned_minus_bestof8"),
                              (circ, pool[:k5], "circuit_minus_bestof128"),
                              (circ, pool_val[:k5], "circuit_minus_bestof128_val"),
                              (uni, pool_val[:k5], "unitary_minus_bestof128_val")):
                mu, se = paired(a, b)                                   # 5 seeds (sweep arms)
                row[f"delta_{lab}"] = mu
                row[f"se_{lab}"] = se
            rows.append(row)
            print(f"  {name:>4} {slab:>5}  weak {row['mean_acc_weak']:.3f}  "
                  f"strong_dft {row['mean_acc_strong_dft']:.3f}  strong_rand {row['mean_acc_strong_random']:.3f}  "
                  f"strong_learned {row['mean_acc_strong_learned']:.3f}  random_full {rnd:.3f}  "
                  f"unitary {row['mean_acc_learned_unitary']:.3f}  circuit {row['mean_acc_learned_circuit']:.3f}  "
                  f"best8 {row['mean_acc_bestof8_fixed']:.3f}  best128 {row['mean_acc_bestof128_fixed']:.3f}  best128val {row['mean_acc_bestof128_val']:.3f}  "
                  f"blocks {row['block_sizes']}  ranks w/sDFT/sRnd/sLrn/full={ranks['weak']}/"
                  f"{ranks['strong_dft']}/{ranks['strong_random']}/{ranks['strong_learned']}/"
                  f"{ranks['random_full']}  bounds {bound_weak}/{bound_strong}/{bound_full}  incl {incl_err:.1e}")

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT, f"({len(rows)} rows)")

    # ---- pooled summary over the nine non-abelian groups ------------------
    srows = []
    for slab in ("exact", "256"):
        na = [r for r in rows if r["shots"] == slab and not r["abelian"]]
        ab = [r for r in rows if r["shots"] == slab and r["abelian"]]
        s = dict(shots=slab, n_nonabelian=len(na), n_abelian=len(ab))
        for k in ("weak", "strong_dft", "strong_random", "strong_learned",
                  "random_full", "learned_unitary", "learned_circuit", "bestof8_fixed", "bestof128_fixed",
                  "bestof128_val", "bestof128_fixed_first5", "bestof128_val_first5"):
            s[f"nonab_mean_{k}"] = float(np.mean([r[f"mean_acc_{k}"] for r in na]))
            s[f"abelian_mean_{k}"] = float(np.mean([r[f"mean_acc_{k}"] for r in ab]))
        for lab in ("strong_dft_minus_weak", "strong_random_minus_weak", "strong_learned_minus_weak",
                    "circuit_minus_strong_learned", "unitary_minus_strong_learned",
                    "circuit_minus_strong_dft", "strong_learned_minus_bestof8",
                    "strong_learned_minus_bestof128", "circuit_minus_bestof128",
                    "strong_learned_minus_bestof128_val", "circuit_minus_bestof128_val",
                    "unitary_minus_bestof128_val", "weak_minus_bestof128_val", "weak_minus_bestof128"):
            d = np.array([r[f"delta_{lab}"] for r in na])
            s[f"nonab_mean_{lab}"] = float(d.mean())
            s[f"nonab_groups_positive_{lab}"] = int((d > 0).sum())
            s[f"nonab_groups_positive_2se_{lab}"] = int(sum(1 for r in na if r[f"delta_{lab}"] > 2 * r[f"se_{lab}"]))
        s["abelian_strong_equals_weak_all"] = int(all(r["strong_equals_weak_abelian"] == 1 for r in ab))
        from scipy import stats as sps
        for lab in ("strong_learned_minus_bestof128", "strong_learned_minus_bestof8", "circuit_minus_bestof128",
                    "strong_learned_minus_bestof128_val", "circuit_minus_bestof128_val"):
            pg = np.array([r[f"delta_{lab}"] for r in na])
            t, pv = sps.ttest_1samp(pg, 0, alternative="greater")
            s[f"pooled_t_{lab}"] = float(t)
            s[f"pooled_p_onesided_{lab}"] = float(pv)
        # abelian controls against each pool, two-sided, at 20 seeds and at the first 5
        for lab in ("strong_learned_minus_bestof128", "strong_learned_minus_bestof128_val", "weak_minus_bestof128",
                    "weak_minus_bestof128_val"):
            s[f"abelian_groups_differ_2se_{lab}"] = int(sum(1 for r in ab if abs(r[f"delta_{lab}"]) > 2 * r[f"se_{lab}"]))
            s[f"abelian_groups_differ_2se5_{lab}"] = int(sum(1 for r in ab if abs(r[f"delta5_{lab}"]) > 2 * r[f"se5_{lab}"]))
            s[f"nonab_groups_positive_2se5_{lab}"] = int(sum(1 for r in na if r[f"delta5_{lab}"] > 2 * r[f"se5_{lab}"]))
        def verdict(lab, n20=True):
            if slab != "256":
                return "n/a (criterion read at 256 shots)"
            pos = s[f"nonab_groups_positive_2se_{lab}"] if n20 else s[f"nonab_groups_positive_2se5_{lab}"]
            dif = s[f"abelian_groups_differ_2se_{lab}"] if n20 else s[f"abelian_groups_differ_2se5_{lab}"]
            return "CONFIRMS" if (pos >= 5 and dif == 0) else "KILLS"
        s["shot_matched_advantage_verdict"] = verdict("strong_learned_minus_bestof128_val", True)
        s["shot_matched_advantage_verdict_5seeds"] = verdict("strong_learned_minus_bestof128_val", False)
        s["verdict_vs_train_selected_pool"] = verdict("strong_learned_minus_bestof128", True)
        s["verdict_vs_train_selected_pool_5seeds"] = verdict("strong_learned_minus_bestof128", False)
        s["strong_ranks_within_bound_all"] = int(all(
            max(r["rank_strong_dft"], r["rank_strong_random"], r["rank_strong_learned"]) <= r["bound_rank_strong"]
            for r in na + ab))
        s["mixing_rank_exceeds_strong_bound_groups"] = int(sum(
            1 for r in na if r["rank_random_full"] > r["bound_rank_strong"]))
        s["nonab_mean_strong_deficit"] = float(np.mean([r["bound_rank_full"] - r["bound_rank_strong"] for r in na]))
        s["max_inclusion_error"] = max((r["inclusion_error"] for r in rows if r["shots"] == slab), default=0.0)
        srows.append(s)
    with open(OUT_SUMMARY, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0].keys()))
        w.writeheader()
        w.writerows(srows)
    print("wrote", OUT_SUMMARY)
    for s in srows:
        print(f"\n=== {s['shots']} readout, pooled over {s['n_nonabelian']} non-abelian groups ===")
        for k in ("weak", "strong_dft", "strong_random", "strong_learned", "random_full",
                  "learned_unitary", "learned_circuit"):
            print(f"  {k:<16} {s[f'nonab_mean_{k}']:.3f}")
        for k in ("bestof8_fixed", "bestof128_fixed", "bestof128_val"):
            print(f"  {k:<16} {s[f'nonab_mean_{k}']:.3f}   (abelian {s[f'abelian_mean_{k}']:.3f}; abelian weak {s['abelian_mean_weak']:.3f})")
        for lab in ("strong_dft_minus_weak", "strong_random_minus_weak", "strong_learned_minus_weak",
                    "circuit_minus_strong_learned", "unitary_minus_strong_learned",
                    "strong_learned_minus_bestof8", "strong_learned_minus_bestof128", "circuit_minus_bestof128",
                    "strong_learned_minus_bestof128_val", "circuit_minus_bestof128_val", "unitary_minus_bestof128_val"):
            print(f"  {lab:<36} {s[f'nonab_mean_{lab}']:+.3f}   positive {s[f'nonab_groups_positive_{lab}']}/9,"
                  f" >2se {s[f'nonab_groups_positive_2se_{lab}']}/9")
        for tag, lab in (("train-selected", "strong_learned_minus_bestof128"), ("validation-selected", "strong_learned_minus_bestof128_val")):
            print(f"  vs {tag:<19} pool: 20 seeds: nonab >2se {s[f'nonab_groups_positive_2se_{lab}']}/9, abelian differ {s[f'abelian_groups_differ_2se_{lab}']}/3,"
                  f" pooled t={s[f'pooled_t_{lab}']:.2f} p={s[f'pooled_p_onesided_{lab}']:.2g}"
                  f"  |  5 seeds: nonab >2se {s[f'nonab_groups_positive_2se5_{lab}']}/9, abelian differ {s[f'abelian_groups_differ_2se5_{lab}']}/3")
        print(f"  control-vs-control (weak - validation pool, identical constructions on abelian G): differ >2se in"
              f" {s['abelian_groups_differ_2se_weak_minus_bestof128_val']}/3 at 20 seeds, {s['abelian_groups_differ_2se5_weak_minus_bestof128_val']}/3 at 5")
        print(f"  SHOT-MATCHED VERDICT vs validation pool: 20 seeds {s['shot_matched_advantage_verdict']} | 5 seeds {s['shot_matched_advantage_verdict_5seeds']}"
              f"   [vs train-selected pool: 20 seeds {s['verdict_vs_train_selected_pool']} | 5 seeds {s['verdict_vs_train_selected_pool_5seeds']}]")
        print(f"  abelian controls: strong == weak exactly: {bool(s['abelian_strong_equals_weak_all'])}"
              f"   max inclusion error {s['max_inclusion_error']:.1e}")
        print(f"  strong-sampling feature ranks within bound sum(d^2-d+1): {bool(s['strong_ranks_within_bound_all'])};"
              f" sector-mixing rank exceeds that bound in {s['mixing_rank_exceeds_strong_bound_groups']}/9 groups;"
              f" mean strong deficit sum(d-1) = {s['nonab_mean_strong_deficit']:.2f}")


if __name__ == "__main__":
    main()
