"""Noise robustness measured against tasks that the channel provably preserves.

The question this module settles: whether noise changes the task itself, whether
labels remain tied to clean or to noisy states, and whether the lower bound still
applies to the noisy ensemble.

The sweep in experiments/noise_robustness/robustness_sweep.py records only
(task, model, strength, N, acc, acc_std). It does not state which state the label
is a function of, and it carries no Bayes ceiling, so a drop in accuracy is
un-attributable: it could be the METHOD failing or the TASK dissolving. This
module separates the two.

WHAT THIS MODULE ADDS
---------------------
1. LABEL CONVENTION IS AN EXPLICIT, RECORDED VARIABLE. Every row carries
   `label_convention` in {clean_state, noisy_state} and `label_defined_on`.
   Both conventions are RUN for both tasks. They are different learning problems:
     clean_state : y is fixed BEFORE the channel. Noise corrupts the evidence,
                   never the ground truth. ("robustness" reading)
     noisy_state : y is recomputed AFTER the channel from the post-channel state
                   itself. ("the noise redefines the task" reading)
   `label_flip_rate_vs_clean` quantifies how far apart the two problems are.

2. A BAYES CEILING FOR THE NOISY ENSEMBLE, per channel/strength/convention.
   Task A: `bayes_shotlimited_acc` is the EXACT Bayes-optimal decision rule for
     the noisy task at the same shot budget the method gets -- exact multinomial
     + binomial likelihoods for all N candidate shifts (the generative model is
     known in closed form because there are only N distinct preparations), a
     uniform prior (the true prior), and a log-sum-exp over the label classes.
     This is optimal, not a heuristic. Its VALUE is Monte-Carlo estimated over
     seeds/samples, so an SE is reported.
   Task A also carries `helstrom_1copy_bayes_acc`: 1/2 + (1/4)||sigma_0-sigma_1||_1
     on the exact class-averaged post-channel states -- a closed-form, measurement
     -free single-copy ceiling.
   Task B: `helstrom_1copy_bayes_acc` is EXACTLY 0.5 and is derived, not measured:
     both classes have ensemble-average state I/d, and a channel is linear, so
     E[N(rho)] agrees between classes and no single-copy strategy can beat chance.
     The informative ceiling there is the two-copy `bayes_shotlimited_acc`, which
     is MC-estimated from a large pool (labelled `mc_estimated` in
     `bayes_method`), NOT exact -- disclosed per integrity rule 1.

3. TASK INFORMATIVENESS, so a fall in accuracy is attributable.
     `tv_class_distributions`  -- (1/2)||pbar_0 - pbar_1||_1 of the exact
                                  class-conditional readout distributions.
     `trace_dist_class_states` -- (1/2)||sigma_0 - sigma_1||_1 of the exact
                                  class-average post-channel states.
     `min_pairwise_stat_sep`   -- min over distinct preparations of the L2 gap
                                  between their exact statistics. When this hits
                                  0 the noisy task has literally ceased to exist
                                  (the channel has collapsed the state family)
                                  and no learner, informed or not, can do better
                                  than the label prior.
     `label_signal_scale`      -- vacuity guard (integrity rule 3). Any cell
                                  below 1e-10 is marked 'vacuous_excluded' in
                                  `vacuity_flag` and is EXCLUDED from the verdict
                                  BEFORE any comparison is made.

4. PHYSICAL, NOT NOMINAL, CHANNEL STRENGTH. Equal `nominal_strength` across
   channels is NOT equal physical disturbance and this module never assumes it
   is. Per channel/strength we record the exact Choi-derived
   `entanglement_fidelity`, `process_infidelity`, `avg_gate_infidelity` and
   `choi_trace_distance`, plus diamond-norm information split  across
   `diamond_lower_bound` / `diamond_upper_bound` (the standard
   ||J||_1 <= ||.||_diamond <= d||J||_1 sandwich) and
   `diamond_numeric_lower_bound` (an explicit maximisation over pure inputs on
   H (x) H, so every value is a CERTIFIED lower bound; the max over all such
   inputs equals the diamond norm). cvxpy is not installed in this environment,
   so no exact SDP value is claimed. `diamond_closed_form` is populated ONLY for
   the two coherent unitary channels, where Aharonov-Kitaev-Nisan gives
   ||U - I||_diamond = 2|sin(theta/2)| exactly, and serves as a cross-check on
   the numeric bound. There is no column that presents a bound as a measurement
   (integrity rule 7).

TASKS
-----
Task A "modef"  : the Mode-F hidden-shift task (the headline advantage), N=8,
                  channel on every frame qubit, finite-shot Hadamard-test
                  features. Label is a property of the PREPARATION (which shift).
Task B "twocopy": the purity (pure vs mixed) task, n=3, channel i.i.d. on every
                  qubit. Label is a property of the STATE, so the two label
                  conventions genuinely diverge here, which is what makes the
                  distinction quantitative.

THREE LEARNERS ON IDENTICAL TEST DATA (so all deltas are paired, rule 5)
------------------------------------------------------------------------
  cqfa_acc     : the method as published -- nearest-centroid, leave-one-out, on
                 m examples (Task A); trained-threshold two-copy reader (Task B).
  informed_acc : the INFORMED COMPARATOR. Same features, same shot budget, but
                 trained on a large independent pool drawn from the SAME noisy
                 ensemble (so it has effectively learned the channel) with a
                 stronger classifier (shrinkage LDA / pool-optimal threshold).
                 Strictly more informed than the method.
  bayes_*_acc  : the ceiling described above.

PRE-REGISTERED CRITERION (two-sided; fixed BEFORE the first run; applied verbatim)
---------------------------------------------------------------------------------
Define, per non-vacuous cell, on paired per-seed values:
    task_degradation D  = bayes_acc(strength=0) - bayes_acc(strength)
    method_shortfall M  = bayes_acc(strength)   - cqfa_acc(strength)
    informed_shortfall I= bayes_acc(strength)   - informed_acc(strength)
and the excess shortfalls relative to the SAME quantity at strength 0:
    dM = M(strength) - M(0)        dI = I(strength) - I(0)

  SURVIVES  if for every non-vacuous cell either dM <= 0.10, or dM exceeds 0.10
            only where the informed comparator degrades comparably (dI >= dM - 0.05).
            That is: CQFA tracks the noisy-task Bayes ceiling, and wherever CQFA
            falls it is because the TASK fell, not the method.
  KILLED    if there exists a non-vacuous cell with dM > 0.10, resolved at
            > 2 paired SE, while dI <= 0.05. That is: the method falls materially
            below the noisy-task Bayes ceiling while an informed learner with the
            same access does not -- the method, not the task, is failing.
  MIXED     if cells satisfying the KILL inequality exist but none is resolved at
            > 2 paired SE.
The per-cell outcome is written to `cell_verdict` and the aggregate to
results/robustness/noisy_task_verdict.csv. The criterion is applied by code
(`_verdict`), not by prose.

WHAT THIS DOES NOT TEST (integrity rule 9; see also the run summary)
-------------------------------------------------------------------
 * Correlated / non-Markovian / crosstalk noise. All ten channels are i.i.d.
   single-qubit and memoryless.
 * Coherent-error accumulation across circuit depth: the channel is applied once
   at the readout boundary, not per layer.
 * Any real-hardware ensemble. This is density-matrix exact simulation only; no
   new QPU runs were made (integrity rule 8).
 * The Hadamard-test overlap features are sampled independently per irrep k,
   matching the published pipeline in noise_resilience.sample_features. A single
   physical experiment would share shots across k, so absolute shot efficiency is
   optimistic for ALL THREE learners equally -- it cannot create or hide a gap
   between them, which is what the criterion tests.
 * Task B's Bayes ceiling is MC-estimated, not closed form (flagged per row in
   `bayes_method`).

Outputs (all NEW files; nothing existing is overwritten):
  results/robustness/noisy_task_redesign.csv    -- one row per task x channel x
                                                   strength x label_convention
  results/robustness/noisy_task_channels.csv    -- physical channel measures
  results/robustness/noisy_task_verdict.csv     -- the pre-registered verdict

Run:
  PYTHONPATH=. .venv314/bin/python -u -m experiments.noise_robustness.robustness_noisy_task
"""
from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from experiments.noise_robustness import noise_lib as NL

I2 = NL.I2
_EPS = 1e-12
VACUITY_FLOOR = 1e-10          # integrity rule 3


# ===========================================================================
# 1. PHYSICAL CHANNEL MEASURES (exact Choi quantities + certified diamond bounds)
# ===========================================================================
def _choi(kraus, d=2):
    """Normalised Choi J(N) = (N (x) I)(|Phi><Phi|), Tr J = 1 for CPTP N."""
    phi = np.zeros(d * d, dtype=complex)
    for i in range(d):
        phi[i * d + i] = 1.0 / math.sqrt(d)
    rho = np.outer(phi, phi.conj())
    out = np.zeros_like(rho)
    for K in kraus:
        E = np.kron(K, np.eye(d))
        out += E @ rho @ E.conj().T
    return out, rho


def _trace_norm(A):
    return float(np.sum(np.abs(np.linalg.eigvalsh((A + A.conj().T) / 2)))) \
        if np.allclose(A, A.conj().T, atol=1e-12) else float(np.sum(np.linalg.svd(A, compute_uv=False)))


def _diamond_numeric_lower(kraus, d=2, restarts=12, seed=0):
    """Certified lower bound on ||N - I||_diamond.

    ||Delta||_diamond = max over PURE |psi> on H (x) H of ||(Delta (x) I)(|psi><psi|)||_1.
    Every feasible |psi> therefore yields a valid lower bound, and the maximum over
    all of them IS the diamond norm. We maximise numerically with restarts; the
    returned number is reported as a LOWER BOUND, never as the diamond norm.
    """
    rng = np.random.default_rng(seed)
    dim = d * d
    Es = [np.kron(K, np.eye(d)) for K in kraus]

    def neg(v):
        z = (v[:dim] + 1j * v[dim:])
        nz = np.linalg.norm(z)
        if nz < 1e-12:
            return 0.0
        z = z / nz
        rho = np.outer(z, z.conj())
        out = sum(E @ rho @ E.conj().T for E in Es) - rho     # (Delta (x) I)(psi)
        return -float(np.sum(np.linalg.svd(out, compute_uv=False)))

    best = 0.0
    for _ in range(restarts):
        v0 = rng.standard_normal(2 * dim)
        r = minimize(neg, v0, method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-9, fatol=1e-12))
        best = max(best, -float(r.fun))
    return best


def channel_measures(name, strength):
    kraus = NL.NOISE_MODELS[name](strength)
    J, J_id = _choi(kraus)
    # entanglement fidelity F_e = <Phi|J(N)|Phi>, and |Phi><Phi| = J_id, so F_e = Tr(J_id J)
    F_e = float(np.real(np.trace(J_id @ J)))
    dJ = J - J_id
    tn = _trace_norm(dJ)
    d = 2
    closed = ""
    if name in ("coherent_z", "coherent_x"):
        theta = strength * np.pi
        closed = 2.0 * abs(math.sin(theta / 2.0))          # Aharonov-Kitaev-Nisan, exact
    return dict(
        channel=name, nominal_strength=strength,
        cptp_ok=bool(NL.is_trace_preserving(kraus)),
        entanglement_fidelity=F_e,
        process_infidelity=1.0 - F_e,
        avg_gate_infidelity=(d / (d + 1.0)) * (1.0 - F_e),
        choi_trace_distance=0.5 * tn,
        diamond_lower_bound=tn,
        diamond_upper_bound=d * tn,
        diamond_numeric_lower_bound=_diamond_numeric_lower(kraus),
        diamond_closed_form=closed,
        diamond_exact_available=("yes_AKN_unitary" if closed != "" else "no_cvxpy_absent"),
    )


# ===========================================================================
# 2. TASK A -- Mode-F hidden shift.  Exact noisy statistics per preparation.
# ===========================================================================
def _shift(N, g):
    return np.roll(np.eye(N, dtype=complex), g, axis=0)


def _joint(psi, N):
    d = len(psi)
    out = np.zeros(N * d, dtype=complex)
    for g in range(N):
        out[g * d:(g + 1) * d] = _shift(N, g) @ psi
    return out / math.sqrt(N)


def _fourier_povm(N):
    w = np.exp(2j * np.pi / N)
    return [np.array([w ** (k * g) for g in range(N)]) / math.sqrt(N) for k in range(N)]


def modef_exact_stats(phi, N, kraus):
    """For each of the N preparations U_g|phi>, the EXACT post-channel Mode-F
    readout statistics (irrep probabilities p_k and complex overlaps c_k), plus
    the exact post-channel joint state of the psi-branch (for Helstrom).

    Construction matches experiments/fisher_information/noise_resilience.modeF_noisy
    exactly, so this is the ceiling for the PUBLISHED pipeline, not a different one.
    """
    d = N
    nfq = int(round(math.log2(N)))
    phi_r = _joint(phi, N)
    embeds = []
    for q in range(nfq):
        Ks = []
        for K in kraus:
            e = np.array([[1.0 + 0j]])
            for qq in range(nfq):
                e = np.kron(e, K if qq == q else I2)
            Ks.append(np.kron(e, np.eye(d)))
        embeds.append(Ks)

    def chan(mat):
        for Ks in embeds:
            mat = sum(Kk @ mat @ Kk.conj().T for Kk in Ks)
        return mat

    Pks = [np.kron(chi.reshape(-1, 1), np.eye(d)) for chi in _fourier_povm(N)]
    stats = []
    for g in range(N):
        phi_p = _joint(_shift(N, g) @ phi, N)
        r11 = chan(np.outer(phi_p, phi_p.conj()) / 2.0)
        r10 = chan(np.outer(phi_p, phi_r.conj()) / 2.0)
        probs = np.array([2.0 * float(np.real(np.trace(Pk.conj().T @ r11 @ Pk))) for Pk in Pks])
        ov = np.array([2.0 * complex(np.trace(Pk.conj().T @ r10 @ Pk)) for Pk in Pks])
        stats.append(dict(probs=probs, ov=ov, rho=2.0 * r11))     # 2*r11 is the
        # normalised post-channel state of the psi branch (r11 carries the 1/2)
    return stats


def _exact_feature(st):
    return np.concatenate([st["probs"], np.real(st["ov"]), np.imag(st["ov"])])


def _sample_record(st, shots, rng):
    """Shot record, matching noise_resilience.sample_features' generative model."""
    p = np.clip(np.real(st["probs"]), 0, None)
    p = p / p.sum()
    n_k = rng.multinomial(shots, p)
    ov = st["ov"]
    qr = (1.0 + np.clip(np.real(ov), -1, 1)) / 2.0
    qi = (1.0 + np.clip(np.imag(ov), -1, 1)) / 2.0
    a = rng.binomial(shots, qr)
    b = rng.binomial(shots, qi)
    return n_k, a, b


def _record_features(rec, shots):
    n_k, a, b = rec
    return np.concatenate([n_k / shots, 2.0 * a / shots - 1.0, 2.0 * b / shots - 1.0])


def _loglik(rec, st, shots):
    n_k, a, b = rec
    p = np.clip(np.real(st["probs"]), 0, None)
    p = p / p.sum()
    qr = np.clip((1.0 + np.clip(np.real(st["ov"]), -1, 1)) / 2.0, _EPS, 1 - _EPS)
    qi = np.clip((1.0 + np.clip(np.imag(st["ov"]), -1, 1)) / 2.0, _EPS, 1 - _EPS)
    return (float(np.sum(n_k * np.log(p + _EPS)))
            + float(np.sum(a * np.log(qr) + (shots - a) * np.log(1 - qr)))
            + float(np.sum(b * np.log(qi) + (shots - b) * np.log(1 - qi))))


def _bayes_label(rec, stats, labels, shots):
    """EXACT Bayes-optimal label decision for the noisy task at this shot budget:
    uniform prior over the N preparations (the true generative prior), exact
    likelihoods, marginalise within each label class, take the MAP class."""
    L = np.array([_loglik(rec, st, shots) for st in stats])
    l0 = [i for i, y in enumerate(labels) if y == 0]
    l1 = [i for i, y in enumerate(labels) if y == 1]
    s0 = logsumexp(L[l0]) if l0 else -np.inf
    s1 = logsumexp(L[l1]) if l1 else -np.inf
    return int(s1 > s0)


def _nearest_centroid_loo(X, y):
    """The published CQFA readout classifier (copied from fourier_mode so the
    comparison is against the method as published, not a re-tuned variant)."""
    X = np.asarray(X, float); y = np.asarray(y)
    sd = X.std(0)
    keep = sd > 1e-8 * (np.abs(X).mean() + 1e-12)
    maj = max(float(np.mean(y)), 1.0 - float(np.mean(y)))
    if not np.any(keep):
        return maj
    Xs = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
    correct = 0
    for i in range(len(Xs)):
        mask = np.ones(len(Xs), bool); mask[i] = False
        c0 = Xs[mask][y[mask] == 0].mean(0) if np.any(y[mask] == 0) else np.zeros(Xs.shape[1])
        c1 = Xs[mask][y[mask] == 1].mean(0) if np.any(y[mask] == 1) else np.zeros(Xs.shape[1])
        correct += int(int(np.linalg.norm(Xs[i] - c1) < np.linalg.norm(Xs[i] - c0)) == y[i])
    return correct / len(Xs)


def _informed_lda(Xtr, ytr, Xte, yte):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    ytr = np.asarray(ytr); yte = np.asarray(yte)
    if len(np.unique(ytr)) < 2:
        return max(float(np.mean(yte)), 1.0 - float(np.mean(yte)))
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(np.asarray(Xtr, float), ytr)
    return float(np.mean(clf.predict(np.asarray(Xte, float)) == yte))


def _noisy_state_labels(stats, clean_labels, tol=1e-9):
    """Label convention 'noisy_state': the label a fully-informed observer of the
    POST-CHANNEL state assigns. Preparations whose exact post-channel statistics
    coincide to `tol` are indistinguishable in principle and share one label (the
    majority clean label of that group; ties keep the clean label). When the
    channel is injective on the family this reproduces the clean labels exactly --
    which is itself the answer to 'are labels tied to the clean or noisy state'
    for this task, and it is measured, not assumed."""
    F = np.array([_exact_feature(s) for s in stats])
    n = len(stats)
    out = list(clean_labels)
    for i in range(n):
        grp = [j for j in range(n) if np.linalg.norm(F[i] - F[j]) <= tol]
        if len(grp) > 1:
            mean_y = np.mean([clean_labels[j] for j in grp])
            out[i] = clean_labels[i] if abs(mean_y - 0.5) < 1e-12 else int(mean_y > 0.5)
    return out


def _min_pairwise_sep(stats):
    F = np.array([_exact_feature(s) for s in stats])
    n = len(F)
    return float(min(np.linalg.norm(F[i] - F[j]) for i in range(n) for j in range(i + 1, n)))


def run_modef_cell(channel, strength, convention, N, m, shots, pool, seeds):
    """One (channel, strength, convention) cell of Task A, >= 5 seeds, paired."""
    kraus = NL.NOISE_MODELS[channel](strength)
    per_seed = []
    infos = []
    for sd in seeds:
        rng = np.random.default_rng(10_000 + sd)
        v = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        phi = v / np.linalg.norm(v)
        stats = modef_exact_stats(phi, N, kraus)
        clean_labels = [int(g >= N // 2) for g in range(N)]
        labels = clean_labels if convention == "clean_state" \
            else _noisy_state_labels(stats, clean_labels)

        # ---- exact task-informativeness quantities (no sampling) ----
        F = np.array([_exact_feature(s) for s in stats])
        # DEGENERACY GUARD: if the convention collapses every preparation into one
        # class the task no longer exists, and every learner trivially scores the
        # majority baseline. Such a cell must NOT be read as signal: force the
        # label signal to exactly 0 so the vacuity guard excludes it.
        if (0 not in labels) or (1 not in labels):
            label_signal = 0.0
        else:
            c0 = F[[i for i, y in enumerate(labels) if y == 0]].mean(0)
            c1 = F[[i for i, y in enumerate(labels) if y == 1]].mean(0)
            label_signal = float(np.max(np.abs(c1 - c0)))
        p0 = np.mean([stats[i]["probs"] for i, y in enumerate(labels) if y == 0], axis=0) \
            if 0 in labels else np.zeros(N)
        p1 = np.mean([stats[i]["probs"] for i, y in enumerate(labels) if y == 1], axis=0) \
            if 1 in labels else np.zeros(N)
        tv = 0.5 * float(np.sum(np.abs(p1 - p0)))
        s0 = np.mean([stats[i]["rho"] for i, y in enumerate(labels) if y == 0], axis=0) \
            if 0 in labels else np.zeros_like(stats[0]["rho"])
        s1 = np.mean([stats[i]["rho"] for i, y in enumerate(labels) if y == 1], axis=0) \
            if 1 in labels else np.zeros_like(stats[0]["rho"])
        td = 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(s1 - s0))))
        infos.append(dict(label_signal_scale=label_signal, tv_class_distributions=tv,
                    trace_dist_class_states=td,
                    helstrom_1copy_bayes_acc=0.5 + 0.5 * td,
                    min_pairwise_stat_sep=_min_pairwise_sep(stats),
                    label_flip_rate_vs_clean=float(np.mean(
                        [a != b for a, b in zip(labels, clean_labels)])),
                    class_balance=float(np.mean(labels)),
                    majority_baseline=max(float(np.mean(labels)), 1.0 - float(np.mean(labels)))))

        # ---- test set + independent training pool, identical across learners ----
        gs_te = rng.integers(0, N, size=m)
        recs_te = [_sample_record(stats[g], shots, rng) for g in gs_te]
        y_te = [labels[g] for g in gs_te]
        Xte = [_record_features(r, shots) for r in recs_te]

        gs_tr = rng.integers(0, N, size=pool)
        Xtr = [_record_features(_sample_record(stats[g], shots, rng), shots) for g in gs_tr]
        y_tr = [labels[g] for g in gs_tr]

        acc_cqfa = _nearest_centroid_loo(Xte, y_te)
        acc_inf = _informed_lda(Xtr, y_tr, Xte, y_te)
        acc_bayes = float(np.mean([_bayes_label(r, stats, labels, shots) == y
                                   for r, y in zip(recs_te, y_te)]))
        per_seed.append((acc_cqfa, acc_inf, acc_bayes))
    return per_seed, _mean_info(infos)


# ===========================================================================
# 3. TASK B -- purity (pure vs mixed).  Labels here GENUINELY change under noise.
# ===========================================================================
def _haar_pure(n, rng):
    d = 2 ** n
    v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    v /= np.linalg.norm(v)
    return np.outer(v, v.conj())


def _rank2(n, rng):
    d = 2 ** n
    rho = np.zeros((d, d), dtype=complex)
    for _ in range(2):
        v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        v /= np.linalg.norm(v)
        rho += np.outer(v, v.conj())
    return rho / 2.0


def _purity(rho):
    return float(np.real(np.trace(rho @ rho)))


def _swap_shots(pur, T, rng):
    """T destructive-SWAP outcomes on rho^(x)2: each +/-1 with mean = purity."""
    return int(np.sum(rng.random(T) < 0.5 * (1.0 + pur)))     # count of +1


def run_twocopy_cell(channel, strength, convention, n, m, T, pool, seeds, tau=0.75):
    kraus = NL.NOISE_MODELS[channel](strength)
    per_seed = []
    infos = []
    for sd in seeds:
        rng = np.random.default_rng(20_000 + sd)

        def draw(k):
            out = []
            for _ in range(k // 2):
                for rho, yc in ((_haar_pure(n, rng), 1), (_rank2(n, rng), 0)):
                    rn = NL.apply_iid(rho, n, kraus)
                    pn = _purity(rn)
                    y = yc if convention == "clean_state" else int(pn > tau)
                    out.append((pn, y, yc))
            return out

        te = draw(m)
        tr = draw(pool)
        flips = float(np.mean([y != yc for _, y, yc in te + tr]))
        pn1 = [p for p, y, _ in tr if y == 1]
        pn0 = [p for p, y, _ in tr if y == 0]
        y_all = [y for _, y, _ in te + tr]
        # DEGENERACY GUARD (same as Task A): under the noisy_state convention a
        # strong channel can push EVERY state below the purity threshold, leaving a
        # single class. Then all learners trivially hit the majority baseline; that
        # is the task ceasing to exist, not robustness. Force signal to 0.
        if (not pn1) or (not pn0) or (0 not in y_all) or (1 not in y_all):
            gap = 0.0
        else:
            gap = abs(float(np.mean(pn1)) - float(np.mean(pn0)))
        infos.append(dict(label_signal_scale=float(gap), tv_class_distributions=float(gap),
                    trace_dist_class_states=0.0,
                    # DERIVED EXACTLY: both class ensembles average to I/d and a
                    # channel is linear, so E[N(rho)] is identical across classes;
                    # no single-copy strategy can exceed chance on this task.
                    helstrom_1copy_bayes_acc=0.5,
                    min_pairwise_stat_sep=float(gap),
                    label_flip_rate_vs_clean=flips,
                    class_balance=float(np.mean([y for _, y, _ in te])),
                    majority_baseline=max(float(np.mean([y for _, y, _ in te])),
                                          1.0 - float(np.mean([y for _, y, _ in te])))))

        cnt_te = [_swap_shots(p, T, rng) for p, _, _ in te]
        y_te = [y for _, y, _ in te]
        cnt_tr = [_swap_shots(p, T, rng) for p, _, _ in tr]
        y_tr = [y for _, y, _ in tr]

        # CQFA as published: threshold fit on the same m examples (LOO).
        def thr_acc(train_c, train_y, test_c, test_y):
            cand = sorted(set(train_c))
            if not cand or len(set(train_y)) < 2:
                return max(float(np.mean(test_y)), 1.0 - float(np.mean(test_y)))
            best, bt = -1.0, cand[0]
            for t in cand:
                a = np.mean([(int(c > t) == y) for c, y in zip(train_c, train_y)])
                if a > best:
                    best, bt = a, t
            return float(np.mean([(int(c > bt) == y) for c, y in zip(test_c, test_y)]))

        corr = 0
        for i in range(len(cnt_te)):
            msk = [j for j in range(len(cnt_te)) if j != i]
            corr += int(thr_acc([cnt_te[j] for j in msk], [y_te[j] for j in msk],
                                [cnt_te[i]], [y_te[i]]) == 1.0)
        acc_cqfa = corr / len(cnt_te)
        # INFORMED comparator: threshold optimised on the large independent pool.
        acc_inf = thr_acc(cnt_tr, y_tr, cnt_te, y_te)
        # BAYES: MC-estimated mixture likelihood over the pool of noisy purities.
        # (NOT closed form -- flagged as mc_estimated in bayes_method.)
        pool_p1 = np.array(pn1) if pn1 else np.array([0.0])
        pool_p0 = np.array(pn0) if pn0 else np.array([0.0])
        ks = np.arange(T + 1)
        from scipy.stats import binom

        def mix_ll(pool_p):
            return np.log(np.mean(binom.pmf(ks[:, None], T,
                                            np.clip(0.5 * (1 + pool_p[None, :]), _EPS, 1 - _EPS)),
                                  axis=1) + _EPS)
        L1, L0 = mix_ll(pool_p1), mix_ll(pool_p0)
        pri1 = float(np.mean(y_tr)) if len(y_tr) else 0.5
        dec = (L1 + math.log(max(pri1, _EPS))) > (L0 + math.log(max(1 - pri1, _EPS)))
        acc_bayes = float(np.mean([int(dec[c]) == y for c, y in zip(cnt_te, y_te)]))
        per_seed.append((acc_cqfa, acc_inf, acc_bayes))
    return per_seed, _mean_info(infos)


# ===========================================================================
# 4. DRIVER + the pre-registered verdict, applied by code
# ===========================================================================
def _mse(a):
    a = np.asarray(a, float)
    return float(np.mean(a)), (float(np.std(a, ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0)


def _mean_info(infos):
    """Average the per-seed task-informativeness dicts.

    NOTE (bug fixed after the v1 run): these quantities are per-seed random
    variables -- the state family phi is redrawn per seed in Task A and the whole
    ensemble is redrawn per seed in Task B -- so a single seed's value is not the
    cell's value. v1 (results/robustness/noisy_task_redesign.csv) reported the
    LAST seed only for every informativeness column, which understated the label
    flip rate. v1 is retained unmodified; v2 is the corrected file.
    """
    keys = infos[0].keys()
    return {k: float(np.mean([i[k] for i in infos])) for k in keys}


def _verdict(rows):
    """Apply the pre-registered criterion verbatim. Vacuous cells excluded FIRST."""
    base = {}
    for r in rows:
        if float(r["nominal_strength"]) == 0.0:
            base[(r["task"], r["channel"], r["label_convention"])] = r
    kill, mixed = [], []
    for r in rows:
        if r["vacuity_flag"] != "ok":
            r["cell_verdict"] = r["vacuity_flag"]; continue
        if float(r["nominal_strength"]) == 0.0:
            r["cell_verdict"] = "reference_strength0"; continue
        b = base.get((r["task"], r["channel"], r["label_convention"]))
        if b is None:
            r["cell_verdict"] = "no_reference"; continue
        M = float(r["bayes_acc"]) - float(r["cqfa_acc"])
        I = float(r["bayes_acc"]) - float(r["informed_acc"])
        M0 = float(b["bayes_acc"]) - float(b["cqfa_acc"])
        I0 = float(b["bayes_acc"]) - float(b["informed_acc"])
        dM, dI = M - M0, I - I0
        se = float(r["gap_bayes_minus_cqfa_se"])
        r["excess_method_shortfall_dM"] = dM
        r["excess_informed_shortfall_dI"] = dI
        if dM > 0.10 and dI <= 0.05:
            if se > 0 and dM > 2 * se:
                r["cell_verdict"] = "KILL"; kill.append(r)
            else:
                r["cell_verdict"] = "kill_inequality_unresolved"; mixed.append(r)
        else:
            r["cell_verdict"] = "SURVIVES"
    if kill:
        return "KILLED", kill, mixed
    if mixed:
        return "MIXED", kill, mixed
    return "SURVIVES", kill, mixed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--m", type=int, default=40)
    ap.add_argument("--shots", type=int, default=500)
    ap.add_argument("--copies", type=int, default=64)
    ap.add_argument("--pool", type=int, default=400)
    ap.add_argument("--strengths", type=float, nargs="+",
                    default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--out", default="results/robustness")
    ap.add_argument("--tag", default="",
                    help="filename suffix. Existing result files are NEVER overwritten; "
                         "a corrected run must use a new tag (e.g. --tag _v2).")
    args = ap.parse_args(argv)
    assert len(args.seeds) >= 5, "integrity rule 5: >= 5 seeds per reported cell"
    os.makedirs(args.out, exist_ok=True)
    for _f in ("noisy_task_channels", "noisy_task_redesign", "noisy_task_verdict"):
        _p = os.path.join(args.out, f"{_f}{args.tag}.csv")
        if os.path.exists(_p):
            raise SystemExit(f"REFUSING TO OVERWRITE existing result file: {_p}\n"
                             f"Pass a new --tag. (project rule: never overwrite results)")

    channels = list(NL.NOISE_MODELS)

    # ---- physical channel measures ----
    chan_rows = []
    for c in channels:
        for s in args.strengths:
            chan_rows.append(channel_measures(c, s))
            print(f"[chan] {c:<18} s={s:<5} F_e={chan_rows[-1]['entanglement_fidelity']:.4f} "
                  f"diamond in [{chan_rows[-1]['diamond_numeric_lower_bound']:.4f}, "
                  f"{chan_rows[-1]['diamond_upper_bound']:.4f}]")
    p_ch = os.path.join(args.out, f"noisy_task_channels{args.tag}.csv")
    with open(p_ch, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(chan_rows[0].keys())); w.writeheader(); w.writerows(chan_rows)
    print("wrote", p_ch)
    cm = {(r["channel"], r["nominal_strength"]): r for r in chan_rows}

    # ---- main sweep ----
    rows = []
    for task in ("modef", "twocopy"):
        for c in channels:
            for s in args.strengths:
                for conv in ("clean_state", "noisy_state"):
                    if task == "modef":
                        per, info = run_modef_cell(c, s, conv, args.N, args.m,
                                                   args.shots, args.pool, args.seeds)
                        sz, budget, bmeth = args.N, args.shots, "exact_map_known_generative_model"
                    else:
                        per, info = run_twocopy_cell(c, s, conv, args.n, args.m,
                                                     args.copies, args.pool, args.seeds)
                        sz, budget, bmeth = args.n, args.copies, "mc_estimated_from_pool"
                    cq = [p[0] for p in per]; inf = [p[1] for p in per]; by = [p[2] for p in per]
                    d_bc = [b - a for b, a in zip(by, cq)]      # paired: diff first
                    d_bi = [b - a for b, a in zip(by, inf)]
                    d_ic = [a - b for a, b in zip(inf, cq)]
                    mc, sc = _mse(cq); mi, si = _mse(inf); mb, sb = _mse(by)
                    g1, e1 = _mse(d_bc); g2, e2 = _mse(d_bi); g3, e3 = _mse(d_ic)
                    ch = cm[(c, s)]
                    rows.append(dict(
                        task=task, channel=c, nominal_strength=s,
                        label_convention=conv,
                        label_defined_on=("clean_state_pre_channel" if conv == "clean_state"
                                          else "noisy_state_post_channel"),
                        size=sz, n_seeds=len(args.seeds), m_test=args.m,
                        pool_train=args.pool, measurement_budget=budget,
                        # physical channel strength (NOT the nominal parameter)
                        entanglement_fidelity=ch["entanglement_fidelity"],
                        process_infidelity=ch["process_infidelity"],
                        avg_gate_infidelity=ch["avg_gate_infidelity"],
                        choi_trace_distance=ch["choi_trace_distance"],
                        diamond_lower_bound=ch["diamond_lower_bound"],
                        diamond_upper_bound=ch["diamond_upper_bound"],
                        diamond_numeric_lower_bound=ch["diamond_numeric_lower_bound"],
                        diamond_closed_form=ch["diamond_closed_form"],
                        # task definition / informativeness
                        label_flip_rate_vs_clean=info["label_flip_rate_vs_clean"],
                        class_balance=info["class_balance"],
                        majority_baseline=info["majority_baseline"],
                        label_signal_scale=info["label_signal_scale"],
                        vacuity_flag=("single_class_degenerate"
                                      if info["majority_baseline"] >= 1.0 - 1e-12
                                      else ("vacuous_excluded"
                                            if info["label_signal_scale"] < VACUITY_FLOOR
                                            else "ok")),
                        tv_class_distributions=info["tv_class_distributions"],
                        trace_dist_class_states=info["trace_dist_class_states"],
                        min_pairwise_stat_sep=info["min_pairwise_stat_sep"],
                        helstrom_1copy_bayes_acc=info["helstrom_1copy_bayes_acc"],
                        # learners (paired, same test data)
                        bayes_method=bmeth,
                        bayes_acc=mb, bayes_se=sb,
                        informed_acc=mi, informed_se=si,
                        cqfa_acc=mc, cqfa_se=sc,
                        gap_bayes_minus_cqfa=g1, gap_bayes_minus_cqfa_se=e1,
                        gap_bayes_minus_informed=g2, gap_bayes_minus_informed_se=e2,
                        gap_informed_minus_cqfa=g3, gap_informed_minus_cqfa_se=e3,
                        excess_method_shortfall_dM="", excess_informed_shortfall_dI="",
                        cell_verdict="",
                    ))
                    print(f"[{task}] {c:<18} s={s:<5} {conv:<12} "
                          f"bayes={mb:.3f} inf={mi:.3f} cqfa={mc:.3f} "
                          f"gap(B-C)={g1:+.3f}+/-{e1:.3f} flip={info['label_flip_rate_vs_clean']:.3f} "
                          f"{'VACUOUS' if info['label_signal_scale'] < VACUITY_FLOOR else ''}")

    overall, kill, mixed = _verdict(rows)
    p_main = os.path.join(args.out, f"noisy_task_redesign{args.tag}.csv")
    with open(p_main, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", p_main)

    nonvac = [r for r in rows if r["vacuity_flag"] == "ok"]
    vrow = dict(
        preregistered_criterion=("SURVIVES iff for every non-vacuous cell dM<=0.10 or "
                                 "dI>=dM-0.05; KILLED iff exists a cell with dM>0.10 at "
                                 ">2 paired SE while dI<=0.05; MIXED if such cells exist "
                                 "but none resolved at >2 SE."),
        overall_verdict=overall, n_rows=len(rows), n_nonvacuous=len(nonvac),
        n_vacuous_excluded=len(rows) - len(nonvac),
        n_single_class_degenerate=sum(1 for r in rows
                                      if r["vacuity_flag"] == "single_class_degenerate"),
        n_task_destroyed_by_noise=sum(1 for r in rows if r["vacuity_flag"] != "ok"
                                      and float(r["nominal_strength"]) > 0.0),
        n_kill_cells=len(kill), n_unresolved_kill_cells=len(mixed),
        max_excess_method_shortfall_dM=max(
            [float(r["excess_method_shortfall_dM"]) for r in rows
             if r["excess_method_shortfall_dM"] != ""] or [float("nan")]),
        max_gap_bayes_minus_cqfa=max([r["gap_bayes_minus_cqfa"] for r in nonvac] or [float("nan")]),
        n_seeds=len(args.seeds),
    )
    p_v = os.path.join(args.out, f"noisy_task_verdict{args.tag}.csv")
    with open(p_v, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(vrow.keys())); w.writeheader(); w.writerow(vrow)
    print("wrote", p_v)
    print(f"\n=== PRE-REGISTERED VERDICT: {overall} "
          f"({len(kill)} KILL cells, {len(mixed)} unresolved, "
          f"{len(rows)-len(nonvac)} vacuous excluded) ===")
    return rows


if __name__ == "__main__":
    main()
