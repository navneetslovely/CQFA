
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
   
    keys = infos[0].keys()
    return {k: float(np.mean([i[k] for i in infos])) for k in keys}


def _verdict(rows):
    
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
