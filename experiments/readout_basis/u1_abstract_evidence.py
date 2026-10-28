"""
U1 abstract evidence -- the two measurements the abstract sentence depends on.

WHY THIS FILE EXISTS
--------------------
The adversarial review of U1 found two defects that changed what U1 may claim, and the
corrected numbers currently exist only in scratch scripts. Integrity rule 1 forbids citing a
number that is not in a committed CSV, so this module recomputes both and writes them.

DEFECT 5 -- N_FIXED_BASES = 8 is an arbitrary constant, and the shot-matched headline was an
artifact of it. `modeF_bestbasis` is best-of-8 random generalized Fourier bases. Nothing makes
8 special. Sweeping the candidate pool size K shows that at S=256 a best-of-128 FIXED pool
ties the learned circuit (~-0.000) at less than half the circuit's readout-fitting cost, so
"learning beats fixed readouts at matched shots" is FALSE at that budget. At exact shots the
same attack fails: the fixed pool saturates well below the learned circuit. Part A measures
the whole K-curve at both budgets so the claim can be stated at the budget where it holds and
withdrawn at the budget where it does not.

DEFECT 6 -- the mechanism regressor was wrong. `max_offdiag_after_fourier` in
u1_structure_check.csv is max|entry| of the off-diagonal of F^dag rho F for ONE single random
state (rhos[0], seed 7), unnormalised. It ANTI-correlates with the ensemble off-diagonal
weight it was meant to proxy, which is why the reported mechanism correlation was a null
(r=0.11, p=0.77). Part B replaces it with two proper regressors: the ensemble-averaged
relative off-Fourier-diagonal Frobenius weight, and the closed-form group invariant
    invisible_fraction = 1 - (sum_lambda d_lambda) / |G|
which is the fraction of the frame state's weight lying off the Fourier diagonal for a
non-abelian group (it is exactly 0 for abelian G, where sum d_lambda = |G|).

WHAT IS AND IS NOT A MEASUREMENT HERE
-------------------------------------
The invisible fraction is a GROUP-THEORETIC CONSTANT, not a measurement -- it is computed
from the irrep dimensions. The measurement is the accuracy delta. The claim under test is
that the delta scales with that constant. Reporting it as a correlation over 9 groups is
therefore a test of a structural prediction, not a discovered empirical law, and the paper
must say so.

PRE-REGISTERED, TWO-SIDED (fixed before this file was first run):
  Part A CONFIRMS the learned readout only at budgets where the best-of-K fixed pool
    SATURATES strictly below the learned circuit as K grows. It KILLS the shot-matched claim
    at any budget where a fixed pool reaches parity at or below the circuit's own fitting cost.
  Part B CONFIRMS the mechanism if delta correlates with the invisible fraction with the
    correlation excluding 0 and surviving leave-one-out. It KILLS it if the correlation is
    null, or if it is explained equally well by 1/|G| (i.e. the effect is a size artifact
    rather than a representation-theoretic one) -- so 1/|G| is included as a rival regressor
    and must be beaten, not ignored.

Reads : results/readout_basis/u1_shotmatched_percell.csv  (committed, unchanged)
Writes: results/readout_basis/u1_pool_sweep.csv           (NEW)
        results/readout_basis/u1_mechanism.csv            (NEW)
Nothing is overwritten or deleted.
"""
from __future__ import annotations

import csv
import math
import os

import numpy as np
from scipy import stats as sps

import experiments.readout_basis.u1_shotmatched_nonabelian as U

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "results", "readout_basis")
PERCELL = os.path.join(OUT, "u1_shotmatched_percell.csv")

NONAB = ["S3", "D4", "Q8", "D5", "A4", "D6", "D10", "F21", "S4"]
ABEL = ["C6", "C8", "C12"]
TASK = "generic_W"
SEEDS = [0, 1, 2, 3, 4]
KMAX = 320
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 320]

# irrep dimension multisets; asserted against |G| = sum d^2 below, so a typo cannot pass
IRREP = {"S3": [1, 1, 2], "D4": [1, 1, 1, 1, 2], "Q8": [1, 1, 1, 1, 2], "D5": [1, 1, 2, 2],
         "A4": [1, 1, 1, 3], "D6": [1, 1, 1, 1, 2, 2], "D10": [1, 1, 1, 1, 2, 2, 2, 2],
         "F21": [1, 1, 1, 3, 3], "S4": [1, 1, 2, 3, 3],
         "C6": [1] * 6, "C8": [1] * 8, "C12": [1] * 12}


def _load():
    rows = list(csv.DictReader(open(PERCELL)))
    cells = {}
    for r in rows:
        cells.setdefault((r["group"], r["task"], r["shots"]), []).append(r)
    return cells


def _arr(cells, key, col):
    sub = sorted(cells[key], key=lambda r: int(r["seed"]))
    return np.array([float(r[col]) for r in sub])


def part_a(cells):
    """Best-of-K fixed GFT pool, selected on TRAIN, vs the learned circuit."""
    out = []
    for S, slab in ((256, "256"), (None, "exact")):
        TE, TR = {}, {}
        for name in NONAB:
            T, m = U.group_table(name)
            L, Rop = U.regular_ops(T)
            n = max(1, int(math.ceil(math.log2(m))))
            D = 1 << n
            iu = U._pair_idx(m)
            F_m = U.fourier_basis(L, Rop, 0)
            te = np.zeros((len(SEEDS), KMAX))
            tr = np.zeros((len(SEEDS), KMAX))
            for si, seed in enumerate(SEEDS):
                rhos, y, _ = U.make_dataset(m, L, F_m, TASK, seed)
                Psi = U.phi(rhos, iu)
                Ptr, Pte = Psi[:U.N_TRAIN], Psi[U.N_TRAIN:]
                ytr, yte = y[:U.N_TRAIN], y[U.N_TRAIN:]
                g = np.random.default_rng(20250722 + 911 * seed + 13 * D)
                for b in range(KMAX):
                    Fb = U.fourier_basis(L, Rop, b)
                    Fe = np.eye(D, dtype=complex)
                    Fe[:m, :m] = Fb
                    a, bq, _ = U.fit_logreg(
                        U.sample_shots(U.probs(Ptr, Fe, m, iu), S, g), ytr,
                        U.sample_shots(U.probs(Pte, Fe, m, iu), S, g), yte)
                    tr[si, b] = a
                    te[si, b] = bq
            TR[name], TE[name] = tr, te
            print(f"  [poolsweep S={slab}] {name} done", flush=True)

        circ = np.array([_arr(cells, (g, TASK, slab), "acc_test_learned_circuit") for g in NONAB])
        for K in KS:
            best = np.array([[TE[g][si, int(np.argmax(TR[g][si, :K]))] for si in range(len(SEEDS))]
                             for g in NONAB])
            dc = circ - best
            npass = sum(1 for i in range(len(NONAB))
                        if dc[i].mean() > 2 * dc[i].std(ddof=1) / math.sqrt(len(SEEDS)))
            per_group_mean = dc.mean(1)
            t, p = sps.ttest_1samp(per_group_mean, 0, alternative="greater")
            # shots the fixed search itself costs: K candidates x N_TRAIN states x S shots
            sel_shots = "" if S is None else K * U.N_TRAIN * S
            out.append(dict(
                shots=slab, pool_K=K,
                mean_acc_best_of_K_fixed=float(best.mean()),
                mean_acc_learned_circuit=float(circ.mean()),
                mean_delta_circ_minus_bestofK=float(dc.mean()),
                n_groups_pass_2se=npass, n_groups=len(NONAB),
                pooled_t=float(t), pooled_p_onesided=float(p),
                fixed_search_shots=sel_shots,
                learned_readout_fit_shots=30924800,
                note=("fixed pool ties or beats the learned circuit"
                      if dc.mean() <= 0 else ""),
            ))
    return out


def part_b(cells):
    """Mechanism: does the delta scale with the group-theoretic invisible fraction?"""
    reg = {}
    for name in NONAB + ABEL:
        T, m = U.group_table(name)
        L, Rop = U.regular_ops(T)
        F = U.fourier_basis(L, Rop, 0)
        g = np.random.default_rng(11)
        P = g.normal(size=(2000, m)) + 1j * g.normal(size=(2000, m))
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        rhos = U.frame_states(P, L)
        B = np.einsum("ai,nab,bj->nij", F.conj(), rhos, F)
        dg = np.abs(np.einsum("nii->ni", B)) ** 2
        tot = np.sum(np.abs(B) ** 2, axis=(1, 2))
        rel_off = float(np.mean(1.0 - dg.sum(1) / tot))
        d = IRREP[name]
        assert sum(x * x for x in d) == m, f"{name}: irrep dims inconsistent with |G|"
        reg[name] = dict(order=m, sum_d=sum(d), invisible_fraction=1.0 - sum(d) / m,
                         ensemble_rel_offdiag=rel_off, inv_order=1.0 / m)

    rows = []
    for slab in ("256", "exact"):
        for base in ("modeF_fixed", "modeF_bestbasis"):
            y = np.array([(_arr(cells, (g, TASK, slab), "acc_test_learned_circuit")
                           - _arr(cells, (g, TASK, slab), f"acc_test_{base}")).mean()
                          for g in NONAB])
            for lab, key in (("invisible_fraction", "invisible_fraction"),
                             ("ensemble_rel_offdiag", "ensemble_rel_offdiag"),
                             ("inv_order_RIVAL", "inv_order")):
                x = np.array([reg[g][key] for g in NONAB])
                r, p = sps.pearsonr(x, y)
                rs, ps = sps.spearmanr(x, y)
                # leave-one-out range: the correlation must not hinge on one group
                loo = []
                for i in range(len(NONAB)):
                    mask = np.arange(len(NONAB)) != i
                    loo.append(sps.pearsonr(x[mask], y[mask])[0])
                rows.append(dict(
                    shots=slab, baseline=base, regressor=lab, n_groups=len(NONAB),
                    pearson_r=float(r), pearson_p=float(p),
                    spearman_rho=float(rs), spearman_p=float(ps),
                    loo_r_min=float(min(loo)), loo_r_max=float(max(loo)),
                    is_rival_regressor=int(lab.endswith("RIVAL")),
                ))

    per_group = [dict(group=g, **reg[g]) for g in NONAB + ABEL]
    return rows, per_group


def main():
    cells = _load()
    os.makedirs(OUT, exist_ok=True)

    print("PART A -- fixed-GFT pool sweep")
    a = part_a(cells)
    pa = os.path.join(OUT, "u1_pool_sweep.csv")
    assert not os.path.exists(pa), f"refusing to overwrite {pa}"
    with open(pa, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(a[0].keys()))
        w.writeheader(); w.writerows(a)
    print(f"wrote {pa}")

    print("\nPART B -- mechanism regressors")
    b, per_group = part_b(cells)
    pb = os.path.join(OUT, "u1_mechanism.csv")
    assert not os.path.exists(pb), f"refusing to overwrite {pb}"
    with open(pb, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(b[0].keys()))
        w.writeheader(); w.writerows(b)
    pg = os.path.join(OUT, "u1_mechanism_regressors.csv")
    assert not os.path.exists(pg), f"refusing to overwrite {pg}"
    with open(pg, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_group[0].keys()))
        w.writeheader(); w.writerows(per_group)
    print(f"wrote {pb}\nwrote {pg}")

    print("\n=== pool sweep (the shot-matched claim) ===")
    print(f"{'S':>6} {'K':>5} {'fixed':>8} {'circ-fixed':>11} {'pass':>7} {'p':>10}")
    for r in a:
        print(f"{r['shots']:>6} {r['pool_K']:>5} {r['mean_acc_best_of_K_fixed']:8.4f} "
              f"{r['mean_delta_circ_minus_bestofK']:+11.4f} "
              f"{r['n_groups_pass_2se']:>5}/9 {r['pooled_p_onesided']:10.2e}")
    print("\n=== mechanism (non-rival regressors only) ===")
    for r in b:
        if r["is_rival_regressor"]:
            continue
        print(f"  S={r['shots']:>5} vs {r['baseline']:<16} {r['regressor']:<22} "
              f"r={r['pearson_r']:+.4f} p={r['pearson_p']:.2e} loo[{r['loo_r_min']:+.3f},{r['loo_r_max']:+.3f}]")
    print("\n=== rival regressor 1/|G| (must NOT explain it as well) ===")
    for r in b:
        if not r["is_rival_regressor"]:
            continue
        print(f"  S={r['shots']:>5} vs {r['baseline']:<16} r={r['pearson_r']:+.4f} p={r['pearson_p']:.2e}")


if __name__ == "__main__":
    main()
