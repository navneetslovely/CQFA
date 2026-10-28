
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

_ROOT = os.path.abspath(__file__)
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(
        os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
EPS_DIV = 1e-300

# ---------------------------------------------------------------------------
# group, representation, task
# ---------------------------------------------------------------------------


def shift(N: int, g: int) -> np.ndarray:
    """U_g |x> = |x + g mod N>."""
    return np.roll(np.eye(N, dtype=complex), g, axis=0)


def rep_exact(N: int) -> np.ndarray:
    return np.stack([shift(N, g) for g in range(N)])


def draw_generators(N: int, rng) -> np.ndarray:
    """H_g random Hermitian, spectral norm 1, H_0 = 0. Drawn ONCE per seed; eps only rescales."""
    Hs = [np.zeros((N, N), dtype=complex)]
    for _ in range(1, N):
        A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        H = (A + A.conj().T) / 2.0
        Hs.append(H / np.linalg.norm(H, 2))
    return np.stack(Hs)


def rep_broken(N: int, eps: float, Hs: np.ndarray) -> np.ndarray:
    """Utilde_g = U_g exp(i eps H_g)."""
    out = []
    for g in range(N):
        U = shift(N, g)
        out.append(U if (g == 0 or eps == 0.0) else U @ expm(1j * eps * Hs[g]))
    return np.stack(out)


def draw_chi(N, m, rng):
    v = rng.standard_normal((m, N)) + 1j * rng.standard_normal((m, N))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def power_spectrum(chi: np.ndarray) -> np.ndarray:
    """|chi_hat|^2, rows sum to N. Exactly shift-invariant."""
    return np.abs(np.fft.fft(chi, axis=-1)) ** 2


def label_threshold(N, a, rng, n_cal=20000):
    """Median of the invariant score under an independent calibration draw (balanced classes)."""
    chi = draw_chi(N, n_cal, rng)
    return float(np.median(power_spectrum(chi) @ a))


def make_task(N, m_tr, m_te, seed):
    """Draw everything that does NOT depend on eps. Same draws reused at every eps -> paired."""
    rt = np.random.default_rng(4242 + 7919 * seed)
    Hs = draw_generators(N, rt)
    a = rt.standard_normal(N)
    a = a - a.mean()                       # sum_j |chi_hat_j|^2 = N is constant -> centre a
    tau = label_threshold(N, a, np.random.default_rng(9182 + 31 * seed))
    D = rt.standard_normal((N, 2, N))      # fixed diagonal observables (K = N frames, n_obs = 2)

    rd = np.random.default_rng(555000 + 13 * seed)
    chi_tr, chi_te = draw_chi(N, m_tr, rd), draw_chi(N, m_te, rd)
    s_tr = rd.integers(0, N, size=m_tr)
    s_te = rd.integers(0, N, size=m_te)
    t_tr, t_te = power_spectrum(chi_tr) @ a, power_spectrum(chi_te) @ a
    y_tr, y_te = (t_tr > tau).astype(int), (t_te > tau).astype(int)
    return dict(Hs=Hs, a=a, tau=tau, D=D, chi_tr=chi_tr, chi_te=chi_te, s_tr=s_tr, s_te=s_te,
                y_tr=y_tr, y_te=y_te, t_te=t_te)


def observed_orbit(chi, s, Ub, Uex):
    """psi = Utilde_s chi (data, broken rep); Orb[m,g,i] = (U_g psi)(i) (readout, EXACT rep)."""
    psi = np.einsum("mij,mj->mi", Ub[s], chi)
    return np.einsum("gij,mj->mgi", Uex, psi)


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------


def dft_frames(N):
    """c_k(g) = w^{-kg}/sqrt(N). Orthonormal; w_k = P_k psi (irrep projectors)."""
    w = np.exp(2j * np.pi / N)
    k = np.arange(N)
    return w ** (-np.outer(k, k)) / np.sqrt(N)


def delta_frames(N):
    return np.eye(N, dtype=complex)


def polar(C):
    """Nearest orthonormal frame set (batched over leading axes)."""
    U, _, Vh = np.linalg.svd(C)
    return U @ Vh


def frame_states(C, Orb):
    """w_k = (1/sqrt N) sum_g c_k(g) U_g psi.  C:(...,K,N)  Orb:(m,N,N) -> (...,m,K,N)."""
    N = Orb.shape[-1]
    if C.ndim == 2:
        return np.einsum("kg,mgi->mki", C, Orb) / np.sqrt(N)
    return np.einsum("bkg,mgi->bmki", C, Orb) / np.sqrt(N)


# ---------------------------------------------------------------------------
#  samplers (every reported accuracy comes from these)
# ---------------------------------------------------------------------------


def shotcharged_ortho(W, D, S, rng):
    """Orthonormal frame set: ONE setting, joint multinomial over (k,i), all S shots used.
    sum_{k,i} |w_k(i)|^2 = 1 exactly for any orthonormal basis, so nothing is discarded."""
    m, K, N = W.shape
    q = np.abs(W) ** 2
    qf = q.reshape(m, K * N)
    qf = qf / np.maximum(qf.sum(1, keepdims=True), EPS_DIV)
    cnt = rng.multinomial(S, qf).reshape(m, K, N).astype(float)
    nk = cnt.sum(-1)
    p_hat = nk / S
    num = np.einsum("kjn,mkn->mkj", D, cnt)
    mu_hat = np.where(nk[..., None] > 0, num / np.maximum(nk, 1)[..., None], 0.0)
    return np.concatenate([p_hat, mu_hat.reshape(m, -1)], axis=1)


def shotcharged_postsel(W, D, S, rng):
    """Non-orthonormal frames: K separate experiments, S//K shots each, each shot surviving with
    probability p_k. Zero surviving shots => that frame's features carry no information."""
    m, K, N = W.shape
    n_alloc = max(1, S // K)
    q = np.abs(W) ** 2
    p = q.sum(-1)
    n_sel = rng.binomial(n_alloc, np.clip(p, 0.0, 1.0))
    cond = q / np.maximum(p[..., None], EPS_DIV)
    cond = cond / np.maximum(cond.sum(-1, keepdims=True), EPS_DIV)
    cnt = rng.multinomial(n_sel, cond).astype(float)
    p_hat = n_sel / n_alloc
    num = np.einsum("kjn,mkn->mkj", D, cnt)
    mu_hat = np.where(n_sel[..., None] > 0, num / np.maximum(n_sel, 1)[..., None], 0.0)
    X = np.concatenate([p_hat, mu_hat.reshape(m, -1)], axis=1)
    return X, p


# ---------------------------------------------------------------------------
# smooth differentiable surrogate (TRAINING ONLY -- never used for a reported accuracy)
# ---------------------------------------------------------------------------


def surr_feats(W, D, n_shot_p, n_eff_obs, zp, zo):
    """W:(B,m,K,N); D:(K,n_obs,N) or (B,K,n_obs,N). Gaussian shot noise with frozen normals."""
    B, m, K, N = W.shape
    q = np.abs(W) ** 2
    p = q.sum(-1)
    cond = q / np.maximum(p[..., None], EPS_DIV)
    if D.ndim == 3:
        m1 = np.einsum("kjn,bmkn->bmkj", D, cond)
        m2 = np.einsum("kjn,bmkn->bmkj", D ** 2, cond)
    else:
        m1 = np.einsum("bkjn,bmkn->bmkj", D, cond)
        m2 = np.einsum("bkjn,bmkn->bmkj", D ** 2, cond)
    sd = np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))
    p_hat = p + np.sqrt(np.maximum(p * (1.0 - p), 0.0) / n_shot_p) * zp[None]
    n_eff = n_eff_obs * p
    mu_hat = m1 + sd / np.sqrt(np.maximum(n_eff, 1e-9))[..., None] * zo[None]
    return np.concatenate([p_hat, mu_hat.reshape(B, m, -1)], axis=2)


def surr_loss(X, head, y, lam):
    Xs = (X - X.mean(1, keepdims=True)) / np.maximum(X.std(1, keepdims=True), 1e-9)
    z = np.clip(np.einsum("bmf,bf->bm", Xs, head[:, :-1]) + head[:, -1:], -40, 40)
    return (np.mean(np.log1p(np.exp(z)) - y[None] * z, axis=1)
            + lam * np.sum(head[:, :-1] ** 2, axis=1))


# ---------------------------------------------------------------------------
# linear head
# ---------------------------------------------------------------------------


def _std(X, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean(0), X.std(0)
    return (X - mu) / np.maximum(sd, 1e-9), mu, sd


def _logloss(th, X, y, lam):
    z = np.clip(X @ th[:-1] + th[-1], -40, 40)
    return float(np.mean(np.log1p(np.exp(z)) - y * z) + lam * np.sum(th[:-1] ** 2))


def fit_head(X, y, lam=1e-2):
    Xs, mu, sd = _std(X)
    r = minimize(_logloss, np.zeros(Xs.shape[1] + 1), args=(Xs, y, lam),
                 method="L-BFGS-B", options=dict(maxiter=400))
    return r.x, mu, sd


def acc_head(pack, X, y):
    th, mu, sd = pack
    Xs, _, _ = _std(X, mu, sd)
    return float(np.mean((Xs @ th[:-1] + th[-1] > 0).astype(int) == y))


# ---------------------------------------------------------------------------
# end-to-end training of arms 2, 4, 5 (identical optimiser / restarts / selection protocol)
# ---------------------------------------------------------------------------


def _fd_grad_factory(obj_batch, P, h=1e-5):
    def obj_grad(theta):
        T = np.repeat(theta[None, :], P + 1, axis=0)
        T[1:] += h * np.eye(P)
        v = obj_batch(T)
        return float(v[0]), (v[1:] - v[0]) / h
    return obj_grad


def train_arm(kind, Orb, y, D, S, K, N, zp, zo, lam, restarts, rng, maxiter):
    """Returns (candidate_parameter_list, n_maxiter_hits). kind in {soft, control, soft_unitary}."""
    nfeat = K * (1 + D.shape[1])
    n_obs = D.shape[1]
    n_alloc = max(1, S // K)

    if kind == "soft":
        n_shot_p, n_eff_obs = n_alloc, float(n_alloc)      # post-selected: S//K shots per frame
    else:
        n_shot_p, n_eff_obs = S, float(S)                  # orthonormal: single setting, all S

    def unpack_C(T):
        B = T.shape[0]
        KN = K * N
        C = T[:, :KN].reshape(B, K, N) + 1j * T[:, KN:2 * KN].reshape(B, K, N)
        return C, T[:, 2 * KN:]

    if kind == "soft":
        def obj_batch(T):
            C, head = unpack_C(T)
            C = C / np.maximum(np.linalg.norm(C, axis=2, keepdims=True), 1e-12)
            X = surr_feats(frame_states(C, Orb), D, n_shot_p, n_eff_obs, zp, zo)
            return surr_loss(X, head, y, lam)
    elif kind == "soft_unitary":
        def obj_batch(T):
            C, head = unpack_C(T)
            X = surr_feats(frame_states(polar(C), Orb), D, n_shot_p, n_eff_obs, zp, zo)
            return surr_loss(X, head, y, lam)
    elif kind == "control":
        Wfix = frame_states(dft_frames(N), Orb)[None]       # frames FIXED and exactly equivariant
        def obj_batch(T):
            B = T.shape[0]
            KJN = K * n_obs * N
            Db = T[:, :KJN].reshape(B, K, n_obs, N)
            head = T[:, KJN:]
            X = surr_feats(np.broadcast_to(Wfix, (B,) + Wfix.shape[1:]), Db,
                           n_shot_p, n_eff_obs, zp, zo)
            return surr_loss(X, head, y, lam)
    else:
        raise ValueError(kind)

    inits, structured = [], []
    if kind in ("soft", "soft_unitary"):
        for C0 in (dft_frames(N), delta_frames(N)):         # start AT arm 1 and AT arm 3
            t0 = np.concatenate([C0.real.ravel(), C0.imag.ravel(), np.zeros(nfeat + 1)])
            inits.append(t0)
            structured.append(t0)
        for _ in range(restarts):
            Cr = rng.standard_normal((K, N)) + 1j * rng.standard_normal((K, N))
            Cr /= np.linalg.norm(Cr, axis=1, keepdims=True)
            inits.append(np.concatenate([Cr.real.ravel(), Cr.imag.ravel(),
                                         0.1 * rng.standard_normal(nfeat + 1)]))
    else:
        t0 = np.concatenate([D.ravel(), np.zeros(nfeat + 1)])   # start AT arm 1's observables
        inits.append(t0)
        structured.append(t0)
        for _ in range(restarts):
            inits.append(np.concatenate([rng.standard_normal(K * n_obs * N),
                                         0.1 * rng.standard_normal(nfeat + 1)]))

    P = inits[0].size
    og = _fd_grad_factory(obj_batch, P)
    cands, hits = list(structured), 0
    for t0 in inits:
        r = minimize(og, t0, method="L-BFGS-B", jac=True, options=dict(maxiter=maxiter))
        hits += int(r.status == 1)
        cands.append(r.x)
    return cands, hits


def cand_to_readout(kind, t, K, N, n_obs, D):
    """Decode a parameter vector into (frames, observables) for the sampler."""
    if kind == "control":
        KJN = K * n_obs * N
        return dft_frames(N), t[:KJN].reshape(K, n_obs, N)
    KN = K * N
    C = t[:KN].reshape(K, N) + 1j * t[KN:2 * KN].reshape(K, N)
    if kind == "soft_unitary":
        return polar(C), D
    return C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12), D


# ---------------------------------------------------------------------------
# one (eps, seed) cell
# ---------------------------------------------------------------------------


def run_cell(N, K, eps, seed, task, S, m_tr, m_te, lam, restarts, maxiter, arms):
    Uex = rep_exact(N)
    Ub = rep_broken(N, eps, task["Hs"])
    D = task["D"]
    n_obs = D.shape[1]
    ytr, yte = task["y_tr"], task["y_te"]
    Otr = observed_orbit(task["chi_tr"], task["s_tr"], Ub, Uex)
    Ote = observed_orbit(task["chi_te"], task["s_te"], Ub, Uex)

    r_tr = np.random.default_rng(31000 + 101 * seed)
    r_te = np.random.default_rng(77000 + 101 * seed)
    out = {}

    # ---- arms 1 and 3: fixed frames, head only, orthonormal (single setting, all S shots) ----
    for name, C in (("exact", dft_frames(N)), ("trivial", delta_frames(N))):
        if name not in arms:
            continue
        Xtr = shotcharged_ortho(frame_states(C, Otr), D, S, r_tr)
        Xte = shotcharged_ortho(frame_states(C, Ote), D, S, r_te)
        out["acc_" + name] = acc_head(fit_head(Xtr, ytr, lam), Xte, yte)

    # ---- disclosure: the LITERAL single uniform frame (trivial-irrep projector, Mode A) ----
    if "uniform1" in arms:
        Cu = np.ones((K, N), dtype=complex) / np.sqrt(N)
        Wtr, Wte = frame_states(Cu, Otr), frame_states(Cu, Ote)
        Xtr, _ = shotcharged_postsel(Wtr, D, S, r_tr)
        Xte, pu = shotcharged_postsel(Wte, D, S, r_te)
        out["acc_uniform1"] = acc_head(fit_head(Xtr, ytr, lam), Xte, yte)
        out["psel_mean_uniform1"] = float(np.mean(pu))

    # ---- arms 2, 4, 5: trained end-to-end, selected by  train accuracy ----
    zn = np.random.default_rng(9000 + 17 * seed)
    zp = zn.standard_normal((m_tr, K))
    zo = zn.standard_normal((m_tr, K, n_obs))
    hits_total = 0
    for kind in ("soft", "control", "soft_unitary"):
        if kind not in arms:
            continue
        cands, hits = train_arm(kind, Otr, ytr, D, S, K, N, zp, zo, lam, restarts,
                                np.random.default_rng(555 + 37 * seed), maxiter)
        hits_total += hits
        best = (-1.0, None, None)
        for t in cands:
            C, Dc = cand_to_readout(kind, t, K, N, n_obs, D)
            rr = np.random.default_rng(31000 + 101 * seed)     # same  draw for every cand
            if kind == "soft":
                Xtr, _ = shotcharged_postsel(frame_states(C, Otr), Dc, S, rr)
            else:
                Xtr = shotcharged_ortho(frame_states(C, Otr), Dc, S, rr)
            pack = fit_head(Xtr, ytr, lam)
            a_tr = acc_head(pack, Xtr, ytr)
            if a_tr > best[0]:
                best = (a_tr, (C, Dc), pack)
        (C, Dc), pack = best[1], best[2]
        if kind == "soft":
            Xte, pte = shotcharged_postsel(frame_states(C, Ote), Dc, S, r_te)
            out["psel_mean_soft"] = float(np.mean(pte))
            out["psel_sum_soft"] = float(np.mean(pte.sum(1)))
            out["eff_shots_soft"] = float(max(1, S // K) * np.mean(pte.sum(1)))
        else:
            Xte = shotcharged_ortho(frame_states(C, Ote), Dc, S, r_te)
        out["acc_" + kind] = acc_head(pack, Xte, yte)
        out["acc_" + kind + "_train"] = best[0]
    out["lbfgs_maxiter_hit"] = hits_total

    out["label_signal_scale"] = float(np.std(task["t_te"]))
    out["majority_baseline"] = float(max(yte.mean(), 1 - yte.mean()))
    out["eff_shots_ortho"] = float(S)
    out["n_params_soft"] = 2 * K * N
    out["n_params_control"] = K * n_obs * N
    out["n_params_soft_unitary"] = 2 * K * N
    out["n_params_exact"] = 0
    out["n_params_trivial"] = 0
    return out


# ---------------------------------------------------------------------------
# calibration (arm 1 ONLY -- the task is frozen before any other arm is evaluated)
# ---------------------------------------------------------------------------


def calibrate(args):
    N, K = args.N, args.N
    grid = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    seeds = list(range(args.cal_seeds))
    tasks = {s: make_task(N, args.m_train, args.m_test, s) for s in seeds}
    rows = []
    print("CALIBRATION -- reference arm 'exact' at eps = 0 only. Target band [0.65, 0.85], "
          "pick closest to 0.75.\n")
    for S in grid:
        accs = []
        for s in seeds:
            r = run_cell(N, K, 0.0, s, tasks[s], S, args.m_train, args.m_test,
                         args.lam, 0, 0, arms={"exact"})
            accs.append(r["acc_exact"])
            rows.append(dict(shots=S, seed=s, acc_exact=r["acc_exact"],
                             label_signal_scale=r["label_signal_scale"],
                             majority_baseline=r["majority_baseline"]))
        a = np.array(accs)
        print(f"  S={S:>4}  acc_exact = {a.mean():.4f} +- {a.std(ddof=1)/np.sqrt(len(a)):.4f}"
              f"   (seeds {len(a)})")
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, "u3_calibration.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", p)

    means = {S: float(np.mean([r["acc_exact"] for r in rows if r["shots"] == S])) for S in grid}
    inband = {S: v for S, v in means.items() if 0.65 <= v <= 0.85}
    if not inband:
        print("NO shot budget lands in [0.65, 0.85]. Per protocol: STOP, do not run the sweep.")
        return None
    pick = min(inband, key=lambda S: abs(means[S] - 0.75))
    print(f"CHOSEN (pre-declared rule: closest to 0.75 within band): S = {pick}  "
          f"(acc_exact = {means[pick]:.4f}).  TASK NOW FROZEN.")
    return pick


# ---------------------------------------------------------------------------
# pre-registered criterion
# ---------------------------------------------------------------------------


def evaluate_criterion(rows, eps_list, seeds, out_dir, tag=""):
    def col(sub, k):
        return np.array([r[k] for r in sub], dtype=float)

    crit_rows = []
    verdict = {"soft": dict(spec=False, strict=False, suspect0=False),
               "soft_unitary": dict(spec=False, strict=False, suspect0=False)}
    for eps in eps_list:
        sub = sorted([r for r in rows if r["epsilon"] == eps], key=lambda r: r["seed"])
        if not sub:
            continue
        vac = bool(np.any(col(sub, "label_signal_scale") < 1e-10))
        row = dict(epsilon=eps, n_seeds=len(sub), vacuous_excluded=int(vac),
                   label_signal_scale=float(col(sub, "label_signal_scale").mean()),
                   majority_baseline=float(col(sub, "majority_baseline").mean()))
        for a in ("exact", "soft", "trivial", "control", "soft_unitary", "uniform1"):
            k = "acc_" + a
            if k in sub[0]:
                v = col(sub, k)
                row[k + "_mean"] = float(v.mean())
                row[k + "_sem"] = float(v.std(ddof=1) / np.sqrt(len(v)))
        row["psel_mean_soft"] = float(col(sub, "psel_mean_soft").mean())
        row["eff_shots_soft"] = float(col(sub, "eff_shots_soft").mean())
        row["eff_shots_ortho"] = float(col(sub, "eff_shots_ortho").mean())

        for probe in ("soft", "soft_unitary"):
            beats = {}
            for ref in ("exact", "trivial", "control"):
                d = col(sub, "acc_" + probe) - col(sub, "acc_" + ref)
                sem = float(d.std(ddof=1) / np.sqrt(len(d)))
                row[f"d_{probe}_vs_{ref}_mean"] = float(d.mean())
                row[f"d_{probe}_vs_{ref}_sem"] = sem
                beats[ref] = bool(d.mean() > 2 * sem) and sem > 0
                row[f"beats_{probe}_vs_{ref}"] = int(beats[ref])
            if vac:
                continue
            spec = beats["exact"] and beats["trivial"]
            strict = spec and beats["control"]
            row[f"CONFIRMS_SPEC_{probe}"] = int(spec)
            row[f"CONFIRMS_STRICT_{probe}"] = int(strict)
            if eps > 0:
                verdict[probe]["spec"] |= spec
                verdict[probe]["strict"] |= strict
            if eps == 0.0 and beats["exact"]:
                verdict[probe]["suspect0"] = True
        crit_rows.append(row)

    p = os.path.join(out_dir, f"u3_criterion{tag}.csv")
    keys = sorted({k for r in crit_rows for k in r})
    keys = ["epsilon"] + [k for k in keys if k != "epsilon"]
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(crit_rows)
    return crit_rows, verdict, p


# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--eps", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.4, 0.8, 1.6])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--m-train", type=int, default=150)
    ap.add_argument("--m-test", type=int, default=600)
    ap.add_argument("--shots", type=int, default=None, help="frozen by --calibrate")
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--cal-seeds", type=int, default=5)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--auto", action="store_true",
                    help="calibrate, then run the sweep at the shot budget the pre-declared rule "
                         "selects -- no human in the loop between the two stages")
    ap.add_argument("--out", default="results/readout_basis")
    ap.add_argument("--tag", default="", help="suffix for output CSVs (robustness replicates)")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    if args.calibrate or args.auto:
        pick = calibrate(args)
        if not args.auto:
            return
        if pick is None:
            raise SystemExit("calibration found no shot budget in [0.65, 0.85]; protocol says stop")
        args.shots = pick
        print(f"\n>>> TASK FROZEN at S = {pick}. Running all arms now.\n", flush=True)
    if args.shots is None:
        raise SystemExit("--shots must be set from the frozen calibration result")

    N, K = args.N, args.N
    arms = {"exact", "trivial", "uniform1", "soft", "control", "soft_unitary"}
    tasks = {s: make_task(N, args.m_train, args.m_test, s) for s in args.seeds}

    rows, t0 = [], time.time()
    for eps in args.eps:
        for s in args.seeds:
            r = run_cell(N, K, eps, s, tasks[s], args.shots, args.m_train, args.m_test,
                         args.lam, args.restarts, args.maxiter, arms)
            rows.append(dict(epsilon=eps, seed=s, shots=args.shots, N=N, K=K,
                             n_obs=tasks[s]["D"].shape[1],
                             m_train=args.m_train, m_test=args.m_test, **r))
            print(f"eps={eps:<5} seed={s}  exact={r['acc_exact']:.3f} soft={r['acc_soft']:.3f} "
                  f"triv={r['acc_trivial']:.3f} ctrl={r['acc_control']:.3f} "
                  f"softU={r['acc_soft_unitary']:.3f} unif1={r['acc_uniform1']:.3f} | "
                  f"p_sel(soft)={r['psel_mean_soft']:.4f} "
                  f"eff_shots(soft)={r['eff_shots_soft']:.1f} vs {r['eff_shots_ortho']:.0f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    p = os.path.join(args.out, f"u3_soft_equivariance_hard{args.tag}.csv")
    keys = list(rows[0].keys())
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", p)

    crit, verdict, pc = evaluate_criterion(rows, args.eps, args.seeds, args.out, args.tag)
    print("wrote", pc)

    print("\n" + "=" * 118)
    print(f"{'eps':>5} {'exact':>13} {'soft':>13} {'trivial':>13} {'control':>13} "
          f"{'soft_unit':>13} {'d(soft-exact)':>18} {'beats?':>7} {'p_sel':>7}")
    for r in crit:
        f = lambda a: f"{r['acc_'+a+'_mean']:.3f}+-{r['acc_'+a+'_sem']:.3f}"
        print(f"{r['epsilon']:>5} {f('exact'):>13} {f('soft'):>13} {f('trivial'):>13} "
              f"{f('control'):>13} {f('soft_unitary'):>13} "
              f"{r['d_soft_vs_exact_mean']:>+11.4f}+-{r['d_soft_vs_exact_sem']:.4f} "
              f"{str(bool(r['beats_soft_vs_exact'])):>7} {r['psel_mean_soft']:>7.4f}")
    print("=" * 118)
    for probe in ("soft", "soft_unitary"):
        v = verdict[probe]
        tag = "PRIMARY (literal C3)" if probe == "soft" else "CHARITABLE UPPER BOUND"
        print(f"\n[{probe}] {tag}")
        print(f"  CONFIRMS_SPEC   (beats exact AND trivial at some eps>0, >2*se): {v['spec']}")
        print(f"  CONFIRMS_STRICT (also beats parameter-matched control)       : {v['strict']}")
        killed = "KILLED -- never beats the better of exact/trivial anywhere in the sweep"
        print(f"  VERDICT: {'SURVIVES' if v['spec'] else killed}")
        if v["suspect0"]:
            print("  *** SUSPECT_HARNESS: beat the exact arm at eps=0, where the exact arm is "
                  "provably Bayes-optimal. Treat as a bug, not a discovery. ***")
    print(f"\nseeds={len(args.seeds)} eps={args.eps} S={args.shots} N={N} K={K} "
          f"m_train={args.m_train} m_test={args.m_test}")


if __name__ == "__main__":
    main()
