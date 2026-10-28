
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

_ROOT = os.path.abspath(__file__)
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(
        os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------
# group + symmetry-broken data
# ---------------------------------------------------------------------------
def shift(N: int, g: int) -> np.ndarray:
    """U_g |x> = |x + g mod N>  (same convention as experiments/fisher_information)."""
    return np.roll(np.eye(N, dtype=complex), g, axis=0)


def rep_exact(N: int) -> np.ndarray:
    return np.stack([shift(N, g) for g in range(N)])            # (N, N, N)


def rep_broken(N: int, eps: float, rng) -> np.ndarray:
    """Utilde_g = U_g exp(i eps H_g), H_g random Hermitian normalised to spectral norm 1,
    H_0 = 0 (identity stays the identity)."""
    mats = []
    for g in range(N):
        U = shift(N, g)
        if g == 0 or eps == 0.0:
            mats.append(U)
            continue
        A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        H = (A + A.conj().T) / 2.0
        H = H / np.linalg.norm(H, 2)
        mats.append(U @ expm(1j * eps * H))
    return np.stack(mats)


def _rand_state(N, rng):
    v = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    return v / np.linalg.norm(v)


def dataset(N, m, Ub, phi, rng):
    """m examples (psi_s = Utilde_s phi, label 1[s >= N/2])."""
    s = rng.integers(0, N, size=m)
    psi = np.stack([Ub[si] @ phi for si in s])                   # (m, N)
    y = (s >= N // 2).astype(int)
    return psi, y


# ---------------------------------------------------------------------------
# shared machinery
# ---------------------------------------------------------------------------
def orbit(psi_all: np.ndarray, Uex: np.ndarray) -> np.ndarray:
    """Psi[m, g, :] = U_g psi_m  using the EXACT rep (what the readout knows)."""
    return np.einsum("gij,mj->mgi", Uex, psi_all)


def diag_observables(N, rng, n_obs=4):
    """n_obs diagonal (hence co-measurable) observables, entries ~ N(0,1)."""
    return rng.standard_normal((n_obs, N))                       # (n_obs, N) diagonals


def _obs_stats(state, Dg):
    """<M> and shot std sqrt(<M^2>-<M>^2) for diagonal observables Dg on (unnormalised) states.
    state: (..., N) complex; returns mean (..., n_obs), sd (..., n_obs)."""
    prob = np.abs(state) ** 2
    nrm = prob.sum(-1, keepdims=True)
    prob = prob / np.maximum(nrm, 1e-300)
    m1 = prob @ Dg.T
    m2 = prob @ (Dg.T ** 2)
    return m1, np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))


def _noise(sd, n_eff, z):
    """Additive shot noise: sd / sqrt(n_eff) times frozen standard normals z."""
    return sd / np.sqrt(np.maximum(n_eff, 1e-9)) * z


# ---------------------------------------------------------------------------
# arm features (all take frozen standard-normal draws z so the objective is deterministic)
# ---------------------------------------------------------------------------
def feats_A(Psi, Dg, S, z):
    """Mode A: coherent projection onto the trivial irrep, v = (1/N) sum_g U_g psi."""
    v = Psi.mean(1)                                              # (m, N)
    p = np.real(np.sum(np.abs(v) ** 2, -1))                      # post-selection prob
    mu, sd = _obs_stats(v, Dg)
    n_eff = (S * p)[:, None]
    return mu + _noise(sd, n_eff, z), p


def feats_B(Psi, Dg, S, z):
    """Mode B: classical frame averaging (decohered orbit mean). No post-selection."""
    mu_g, sd_g = _obs_stats(Psi, Dg)                              # (m, N, n_obs)
    mu = mu_g.mean(1)
    var = (sd_g ** 2).mean(1) + mu_g.var(1)                       # incl. variance over sampled g
    return mu + _noise(np.sqrt(var), S, z), None


def modeF_components(Psi, N):
    """v_k = (1/N) sum_g w^{-kg} U_g psi   for all irreps k.  sum_k ||v_k||^2 = 1."""
    w = np.exp(-2j * np.pi / N)
    W = w ** (-np.outer(np.arange(N), np.arange(N)))              # (k, g)
    return np.einsum("kg,mgi->mki", W, Psi) / N


def feats_F(Psi, N, ref, S, zp, zph):
    """Mode F: irrep weights p_k plus relative phase of v_k against a fixed reference.
    2 settings -> S/2 shots each; phase precision on sector k scales with (S/2) p_k."""
    V = modeF_components(Psi, N)                                  # (m, k, N)
    p = np.real(np.sum(np.abs(V) ** 2, -1))                       # (m, k), rows sum to 1
    ov = np.einsum("ki,mki->mk", ref.conj(), V)
    ph = np.angle(ov + 1e-300)
    ph_hat = ph + _noise(np.ones_like(p), (S / 2.0) * p, zph)
    p_hat = p + _noise(np.sqrt(np.maximum(p * (1 - p), 0.0)), S / 2.0, zp)
    return np.concatenate([p_hat, np.cos(ph_hat), np.sin(ph_hat)], axis=1)


def feats_C3(Psi, c, Dg, ref_c, S, K, zo, zph):
    """C3: K trainable complex frames.  Frame k prepares sum_g c_kg |g> (normalised),
    the controlled oracle gives sum_g c_kg |g> (x) U_g psi, and post-selecting the frame
    onto |G> leaves the data state w_k = (1/sqrt N) sum_g c_kg U_g psi, so
        p_k = ||sum_g c_kg U_g psi||^2 / N.
    (Uniform c_kg = 1/sqrt(N) reproduces Mode A exactly.)
    2 settings per frame -> S/(2K) shots each, times p_k."""
    N = Psi.shape[-1]
    cn = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    W = np.einsum("kg,mgi->mki", cn, Psi) / np.sqrt(N)            # (m, K, N)
    p = np.real(np.sum(np.abs(W) ** 2, -1))                       # (m, K)
    mu, sd = _obs_stats(W, Dg)                                    # (m, K, n_obs)
    n_eff = (S / (2.0 * K)) * p
    mu_hat = mu + _noise(sd, n_eff[..., None], zo)
    ov = np.einsum("ki,mki->mk", ref_c.conj(), W)
    ph = np.angle(ov + 1e-300)
    ph_hat = ph + _noise(np.ones_like(p), n_eff, zph)
    m = Psi.shape[0]
    return np.concatenate([mu_hat.reshape(m, -1), np.cos(ph_hat), np.sin(ph_hat)], axis=1), p





def _shot_mean(mu, sd, n_sel, z):
    """Estimate of <M> from n_sel post-selected shots; 0 (no information) if n_sel == 0."""
    ok = n_sel > 0
    est = mu + sd / np.sqrt(np.maximum(n_sel, 1)) * z
    return np.where(ok, est, 0.0)


def feats_A_shotcharged(Psi, Dg, S, rng):
    v = Psi.mean(1)
    p = np.real(np.sum(np.abs(v) ** 2, -1))
    mu, sd = _obs_stats(v, Dg)
    n_sel = rng.binomial(S, np.clip(p, 0, 1))                     # (m,)
    z = rng.standard_normal(mu.shape)
    return _shot_mean(mu, sd, n_sel[:, None], z), p


def feats_B_shotcharged(Psi, Dg, S, rng):
    mu_g, sd_g = _obs_stats(Psi, Dg)
    mu = mu_g.mean(1)
    var = (sd_g ** 2).mean(1) + mu_g.var(1)
    return mu + np.sqrt(var / S) * rng.standard_normal(mu.shape), None


def feats_F_shotcharged(Psi, N, ref, S, rng):
    """Mode F, : 2 settings of S//2 shots. Frame measured in the Fourier basis gives a
    MULTINOMIAL over irreps k -> counts; sector k's phase precision is set by its own count."""
    V = modeF_components(Psi, N)
    p = np.real(np.sum(np.abs(V) ** 2, -1))
    p = p / p.sum(1, keepdims=True)
    nA = max(1, S // 2)
    cnt_p = np.stack([rng.multinomial(nA, pi) for pi in p])
    cnt_h = np.stack([rng.multinomial(nA, pi) for pi in p])
    p_hat = cnt_p / nA
    ph = np.angle(np.einsum("ki,mki->mk", ref.conj(), V) + 1e-300)
    ph_hat = ph + rng.standard_normal(ph.shape) / np.sqrt(np.maximum(cnt_h, 1))
    ok = cnt_h > 0
    return np.concatenate([p_hat, np.where(ok, np.cos(ph_hat), 0.0),
                           np.where(ok, np.sin(ph_hat), 0.0)], axis=1)


def feats_C3_shotcharged(Psi, c, Dg, ref_c, S, K, rng):
    """C3, : K frames x 2 settings, S//(2K) shots each, post-selected with prob p_k.
    n_sel ~ Binomial(S//(2K), p_k); n_sel == 0 => that frame returns nothing."""
    N = Psi.shape[-1]
    cn = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    W = np.einsum("kg,mgi->mki", cn, Psi) / np.sqrt(N)
    p = np.real(np.sum(np.abs(W) ** 2, -1))                       # (m, K)
    mu, sd = _obs_stats(W, Dg)
    nA = max(1, S // (2 * K))
    n_obs_sel = rng.binomial(nA, np.clip(p, 0, 1))                # (m, K)
    n_ph_sel = rng.binomial(nA, np.clip(p, 0, 1))
    mu_hat = _shot_mean(mu, sd, n_obs_sel[..., None], rng.standard_normal(mu.shape))
    ph = np.angle(np.einsum("ki,mki->mk", ref_c.conj(), W) + 1e-300)
    ph_hat = ph + rng.standard_normal(ph.shape) / np.sqrt(np.maximum(n_ph_sel, 1))
    ok = n_ph_sel > 0
    m = Psi.shape[0]
    return np.concatenate([mu_hat.reshape(m, -1), np.where(ok, np.cos(ph_hat), 0.0),
                           np.where(ok, np.sin(ph_hat), 0.0)], axis=1), p


# ---------------------------------------------------------------------------
# linear head (logistic regression, L2-regularised, L-BFGS)
# ---------------------------------------------------------------------------
def _standardise(X, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean(0), X.std(0)
    return (X - mu) / np.maximum(sd, 1e-9), mu, sd


def _logloss(theta, X, y, lam):
    z = X @ theta[:-1] + theta[-1]
    z = np.clip(z, -40, 40)
    return float(np.mean(np.log1p(np.exp(z)) - y * z) + lam * np.sum(theta[:-1] ** 2))


def fit_head(Xtr, ytr, lam=1e-2):
    Xs, mu, sd = _standardise(Xtr)
    th = np.zeros(Xs.shape[1] + 1)
    r = minimize(_logloss, th, args=(Xs, ytr, lam), method="L-BFGS-B",
                 options=dict(maxiter=400))
    return r.x, mu, sd


def head_acc(theta, mu, sd, Xte, yte):
    Xs, _, _ = _standardise(Xte, mu, sd)
    pred = (Xs @ theta[:-1] + theta[-1] > 0).astype(int)
    return float(np.mean(pred == yte))


# ---------------------------------------------------------------------------
# C3 end-to-end optimisation (frames + head jointly)
# ---------------------------------------------------------------------------
def fourier_frames(N, K):
    """c_k(g) = w^{-kg} / sqrt(N): the Mode-F frames themselves (C3's charitable init)."""
    w = np.exp(-2j * np.pi / N)
    return np.stack([w ** (-k * np.arange(N)) for k in range(K)]) / np.sqrt(N)


def _unpack(theta, N, K, nfeat):
    c = theta[:K * N].reshape(K, N) + 1j * theta[K * N:2 * K * N].reshape(K, N)
    return c, theta[2 * K * N:]


def _obj_batch(Thetas, Psi, y, Dg, ref_c, S, K, N, zo, zph, lam):
    """Vectorised objective over a BATCH of parameter vectors Thetas (B, P).

    Exactly mirrors feats_C3 + _standardise + _logloss, but with a leading batch axis so
    that a whole finite-difference gradient is one numpy call instead of P Python calls.
    """
    B, P = Thetas.shape
    KN = K * N
    c = Thetas[:, :KN].reshape(B, K, N) + 1j * Thetas[:, KN:2 * KN].reshape(B, K, N)
    head = Thetas[:, 2 * KN:]                                     # (B, nfeat+1)

    cn = c / np.maximum(np.linalg.norm(c, axis=2, keepdims=True), 1e-12)
    W = np.einsum("bkg,mgi->bmki", cn, Psi) / np.sqrt(N)          # (B, m, K, N)
    aw = np.abs(W) ** 2
    p = aw.sum(-1)                                                # (B, m, K)
    prob = aw / np.maximum(p[..., None], 1e-300)
    m1 = prob @ Dg.T
    m2 = prob @ (Dg.T ** 2)
    sd = np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))
    n_eff = (S / (2.0 * K)) * p
    inv = 1.0 / np.sqrt(np.maximum(n_eff, 1e-9))
    mu_hat = m1 + sd * inv[..., None] * zo[None]
    ov = np.einsum("ki,bmki->bmk", ref_c.conj(), W)
    ph_hat = np.angle(ov + 1e-300) + inv * zph[None]
    m = Psi.shape[0]
    X = np.concatenate([mu_hat.reshape(B, m, -1), np.cos(ph_hat), np.sin(ph_hat)], axis=2)

    Xs = (X - X.mean(1, keepdims=True)) / np.maximum(X.std(1, keepdims=True), 1e-9)
    z = np.clip(np.einsum("bmf,bf->bm", Xs, head[:, :-1]) + head[:, -1:], -40, 40)
    return (np.mean(np.log1p(np.exp(z)) - y[None] * z, axis=1)
            + lam * np.sum(head[:, :-1] ** 2, axis=1))


def train_C3(Psi_tr, ytr, Dg, ref_c, S, K, N, zo, zph, lam=1e-2, restarts=None, rng=None):
    nfeat = K * Dg.shape[0] + 2 * K
    args = (Psi_tr, ytr, Dg, ref_c, S, K, N, zo, zph, lam)

    def obj(theta):
        return float(_obj_batch(theta[None, :], *args)[0])

    def obj_grad(theta):
        """Forward-difference gradient, all P partials in ONE batched evaluation."""
        P = theta.size
        h = 1e-5
        T = np.repeat(theta[None, :], P + 1, axis=0)
        T[1:] += h * np.eye(P)
        v = _obj_batch(T, *args)
        return float(v[0]), (v[1:] - v[0]) / h

    inits = []
    c0 = fourier_frames(N, K)                                    # charitable: start at Mode F
    inits.append(np.concatenate([c0.real.ravel(), c0.imag.ravel(), np.zeros(nfeat + 1)]))
    cu = np.ones((K, N), dtype=complex) / np.sqrt(N)             # start at Mode A
    inits.append(np.concatenate([cu.real.ravel(), cu.imag.ravel(), np.zeros(nfeat + 1)]))
    for _ in range(restarts):
        cr = rng.standard_normal((K, N)) + 1j * rng.standard_normal((K, N))
        cr /= np.linalg.norm(cr, axis=1, keepdims=True)
        inits.append(np.concatenate([cr.real.ravel(), cr.imag.ravel(),
                                     0.1 * rng.standard_normal(nfeat + 1)]))

    # Return ALL candidates (plus the un-optimised structured inits). The caller selects by
    #  train accuracy: the smooth surrogate is biased towards the p_sel -> 0 loophole,
    # so selecting on it alone would sabotage C3. This keeps C3 maximally charitable.
    cands = [_unpack(t0, N, K, nfeat)[0] for t0 in inits[:2]]      # Mode-F and Mode-A frames
    for t0 in inits:
        r = minimize(obj_grad, t0, method="L-BFGS-B", jac=True, options=dict(maxiter=300))
        cands.append(_unpack(r.x, N, K, nfeat)[0])
    return cands


# ---------------------------------------------------------------------------
# one (epsilon, seed) cell
# ---------------------------------------------------------------------------
def run_cell(N, eps, seed, m_tr, m_te, S, K, n_obs, restarts):
    rng = np.random.default_rng(100000 + 977 * seed + int(round(eps * 1000)))
    rng_task = np.random.default_rng(4242 + seed)                # task/rep fixed per seed

    Uex = rep_exact(N)
    Ub = rep_broken(N, eps, rng_task)
    phi = _rand_state(N, rng_task)
    Dg = diag_observables(N, rng_task, n_obs)

    psi_tr, ytr = dataset(N, m_tr, Ub, phi, rng)
    psi_te, yte = dataset(N, m_te, Ub, phi, rng)
    Ptr, Pte = orbit(psi_tr, Uex), orbit(psi_te, Uex)

    # reference states for the interference (phase) features: the unshifted base state
    ref_F = modeF_components(orbit(phi[None, :], Uex), N)[0]      # (k, N)
    ref_C = np.einsum("kg,gi->ki", fourier_frames(N, K), (Uex @ phi)) / np.sqrt(N)

    # frozen shot-noise draws for the SMOOTH TRAINING SURROGATE only (C3's objective must be
    # deterministic and differentiable). All REPORTED accuracies use the sampler.
    zn = np.random.default_rng(9000 + seed)
    zCo_tr = zn.standard_normal((m_tr, K, n_obs))
    zCh_tr = zn.standard_normal((m_tr, K))

    # independent -measurement RNGs for train-head-fitting and for test
    r_tr = np.random.default_rng(31000 + seed)
    r_te = np.random.default_rng(77000 + seed)

    out = {}

    XA_tr, _ = feats_A_shotcharged(Ptr, Dg, S, r_tr)
    XA_te, pA_te = feats_A_shotcharged(Pte, Dg, S, r_te)
    th, mu, sd = fit_head(XA_tr, ytr)
    out["acc_A"] = head_acc(th, mu, sd, XA_te, yte)

    XB_tr, _ = feats_B_shotcharged(Ptr, Dg, S, r_tr)
    XB_te, _ = feats_B_shotcharged(Pte, Dg, S, r_te)
    th, mu, sd = fit_head(XB_tr, ytr)
    out["acc_B"] = head_acc(th, mu, sd, XB_te, yte)

    XF_tr = feats_F_shotcharged(Ptr, N, ref_F, S, r_tr)
    XF_te = feats_F_shotcharged(Pte, N, ref_F, S, r_te)
    th, mu, sd = fit_head(XF_tr, ytr)
    out["acc_F_head"] = head_acc(th, mu, sd, XF_te, yte)

    # C3 frames trained on the smooth surrogate (charitable), SELECTED by  train
    # accuracy, then evaluated  on the held-out test set.
    cands = train_C3(Ptr, ytr, Dg, ref_C, S, K, N, zCo_tr, zCh_tr,
                     restarts=restarts, rng=np.random.default_rng(555 + seed))
    best_c, best_tr, best_pack = None, -1.0, None
    for c in cands:
        Xc_tr, _ = feats_C3_shotcharged(Ptr, c, Dg, ref_C, S, K,
                                   np.random.default_rng(31000 + seed))
        th, mu, sd = fit_head(Xc_tr, ytr)
        a_tr = head_acc(th, mu, sd, Xc_tr, ytr)
        if a_tr > best_tr:
            best_tr, best_c, best_pack = a_tr, c, (th, mu, sd)
    th, mu, sd = best_pack
    XC_te, pC_te = feats_C3_shotcharged(Pte, best_c, Dg, ref_C, S, K, r_te)
    out["acc_C3"] = head_acc(th, mu, sd, XC_te, yte)
    out["acc_C3_train"] = best_tr

    out["postsel_prob"] = float(np.mean(pC_te))
    out["postsel_prob_A"] = float(np.mean(pA_te))
    return out


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    ap.add_argument("--m-train", type=int, default=160)
    ap.add_argument("--m-test", type=int, default=400)
    # The shot budget is FIXED across arms within a run, and swept so the comparison is
    # not read off a single hand-picked budget (at very large S every arm saturates at 1.0
    # and the test cannot discriminate at all).
    ap.add_argument("--shots", type=int, nargs="+", default=[64, 256, 1024, 4096])
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n-obs", type=int, default=4)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--out", default="results/readout_basis")
    args = ap.parse_args(argv)

    rows = []
    for S in args.shots:
        for eps in args.eps:
            for sd in args.seeds:
                r = run_cell(args.N, eps, sd, args.m_train, args.m_test, S,
                             args.K, args.n_obs, args.restarts)
                rows.append(dict(epsilon=eps, seed=sd, shots=S, **r))
                print(f"S={S:<5} eps={eps:<5} seed={sd}  A={r['acc_A']:.3f}  B={r['acc_B']:.3f}  "
                      f"F+head={r['acc_F_head']:.3f}  C3={r['acc_C3']:.3f}  "
                      f"p_sel(C3)={r['postsel_prob']:.4f}  p_sel(A)={r['postsel_prob_A']:.4f}",
                      flush=True)

    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, "t3_soft_equivariance_epsilon.csv")
    order = ["epsilon", "seed", "shots", "acc_A", "acc_B", "acc_F_head", "acc_C3",
             "postsel_prob", "postsel_prob_A", "acc_C3_train"]
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=order)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", p)

    # ---- pre-registered kill criterion -----------------------------------
    n_seeds = len(args.seeds)
    any_beat = False
    for S in args.shots:
        print(f"\n--- shot budget S = {S} per test example ---")
        print(f"{'eps':>6} {'A':>16} {'B':>16} {'F+head':>16} {'C3':>16} "
              f"{'C3-F':>18} {'2*SEM':>8} {'beats?':>7}  {'p_sel(C3)':>10}")
        for eps in args.eps:
            sub = [r for r in rows if r["epsilon"] == eps and r["shots"] == S]
            g = lambda k: np.array([r[k] for r in sub])
            d = g("acc_C3") - g("acc_F_head")
            sem = float(np.std(d, ddof=1) / np.sqrt(n_seeds))
            beat = bool(d.mean() > 2 * sem) and sem > 0
            any_beat |= beat
            f = lambda k: f"{g(k).mean():.3f}+-{g(k).std(ddof=1):.3f}"
            print(f"{eps:>6} {f('acc_A'):>16} {f('acc_B'):>16} {f('acc_F_head'):>16} "
                  f"{f('acc_C3'):>16} {d.mean():>+10.4f}+-{sem:.4f} {2*sem:>8.4f} "
                  f"{str(beat):>7}  {g('postsel_prob').mean():>10.4f}")
    print(f"\nseeds = {n_seeds} ({args.seeds}); N = {args.N}; K = {args.K}; "
          f"m_train = {args.m_train}; m_test = {args.m_test}")
    print("PRE-REGISTERED KILL CRITERION: C3 survives iff mean(acc_C3 - acc_F_head) > 2*SEM "
          "at some epsilon.")
    print("VERDICT:", "C3 SURVIVES" if any_beat else "C3 IS DEAD (no epsilon passes)")


if __name__ == "__main__":
    main()
