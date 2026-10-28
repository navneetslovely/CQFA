"""
experiments.trainability_law.analysis
===============================
Shared, deterministic estimators for the trainability-scaling experiment:

  * wide_params        -- the wide/Haar-like parameter init required for a
                          barren-plateau test (NOT the near-identity training
                          init, which would understate the plateau).
  * random_pure_state  -- a reproducible random pure state.
  * gradient_variance  -- Var_theta[ d_mu f ] over S random parameter draws at
                          a FIXED input, for a given model and component mu.
  * bootstrap_ci       -- percentile bootstrap CI of the variance.
  * fit_exp / fit_poly -- log-linear decay fits with R^2 (plateau vs trainable).
  * classify_decay     -- the plateau | trainable | inconclusive decision rule.

All randomness flows through np.random.default_rng(seed).  Reuses the exact
parameter-shift gradients from cqfa.simulator via Model.features_batch (we do
NOT reimplement gradients).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# parameter / state sampling
# ---------------------------------------------------------------------------
def wide_params(P: int, rng: np.random.Generator) -> np.ndarray:
    """theta ~ Uniform[0, 2*pi)^P.

    The barren-plateau definition randomises theta over the WIDE distribution
    (approximately Haar over the ansatz), NOT the N(0, 0.1) near-identity
    training init used by Model.init_params -- that init sits in a locally
    large-gradient region and would falsely report "no plateau".
    """
    return rng.uniform(0.0, 2.0 * np.pi, size=P)


def random_pure_state(dim: int, rng: np.random.Generator) -> np.ndarray:
    """A reproducible Haar-random pure state on C^dim."""
    v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return v / np.linalg.norm(v)


def fixed_input_for(model, rep, rng: np.random.Generator,
                    p_min: float = 1e-6, max_tries: int = 200) -> np.ndarray:
    """A fixed input state held constant across the S theta-draws.

    For the CQFA model the state must not be annihilated by the coherent
    projector Pi (||Pi psi||^2 >= p_min); we resample until it isn't.  For the
    other models any pure state is fine.
    """
    Pi = rep.projector()
    for _ in range(max_tries):
        psi = random_pure_state(rep.dim, rng)
        if getattr(model, "name", "") == "cqfa":
            phi = Pi @ psi
            if float(np.real(phi.conj() @ phi)) >= p_min:
                return psi
        else:
            return psi
    return psi  # give up gracefully; caller records the (rare) low-p case


# ---------------------------------------------------------------------------
# gradient-variance estimator (the barren-plateau money quantity)
# ---------------------------------------------------------------------------
def component_indices(n_qubits: int, n_layers: int) -> dict:
    """Flat parameter indices for the three reported components.

    Layout (from cqfa.simulator.Ansatz): theta reshaped to
    (n_layers, n_qubits, 2) = (RY, RZ) per qubit per layer, flattened, so the
    RY on qubit q in layer l sits at flat index ((l*n_qubits + q)*2).
    """
    early = ((0 * n_qubits + 0) * 2)
    mid = ((max(0, n_layers // 2) * n_qubits + 0) * 2)
    return {"early": int(early), "mid": int(mid)}


def gradient_samples(model, psi: np.ndarray, S: int, rng: np.random.Generator):
    """Return the (S, P) matrix of parameter-shift gradients d f / d theta for
    S wide-random theta draws at the FIXED input `psi`.

    Uses Model.features_batch (exact parameter-shift), batching all S draws.
    """
    P = model.n_params
    Thetas = np.array([wide_params(P, rng) for _ in range(S)])
    X = np.tile(psi[None, :], (1, 1))                 # (1, dim)
    grads = np.zeros((S, P))
    for s in range(S):
        _, g = model.features_batch(Thetas[s], X)     # g shape (1, P)
        grads[s] = g[0]
    return grads


def gradient_variance(model, psi, S, rng, mu_index: int) -> tuple[float, np.ndarray]:
    """Var_theta[ d_mu f ] at component `mu_index`, plus the S raw gradient
    values (for bootstrap CIs).  A single-sample variance estimate.
    """
    grads = gradient_samples(model, psi, S, rng)
    vals = grads[:, mu_index]
    return float(np.var(vals)), vals


def gradient_variance_all(model, psi, S, rng, mu_dict: dict):
    """Variance at each named component in `mu_dict` plus the max over all
    components.  Returns {name: (var, raw_vals)} including key 'max'.

    Computes the full (S, P) gradient matrix ONCE and slices it, so the three
    components share the same S theta-draws (cheap and consistent).
    """
    grads = gradient_samples(model, psi, S, rng)
    out = {}
    for name, mu in mu_dict.items():
        out[name] = (float(np.var(grads[:, mu])), grads[:, mu])
    per_comp_var = np.var(grads, axis=0)
    mu_max = int(np.argmax(per_comp_var))
    out["max"] = (float(per_comp_var[mu_max]), grads[:, mu_max])
    return out


# ---------------------------------------------------------------------------
# bootstrap CI
# ---------------------------------------------------------------------------
def bootstrap_var_ci(vals, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the VARIANCE of `vals` (resampling the raw
    gradient values, recomputing Var each resample).
    """
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return float(np.var(vals)), float(np.var(vals))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(vals), size=len(vals))
        boot[b] = np.var(vals[idx])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_slope_ci(xs, ys, fit_fn, n_boot=2000, alpha=0.05, seed=0):
    """Bootstrap CI of a fitted slope by resampling the (x, y) sweep points."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 3:
        s, _, _ = fit_fn(xs, ys)
        return float(s), float(s)
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(xs), size=len(xs))
        if len(np.unique(xs[idx])) < 2:
            continue
        s, _, _ = fit_fn(xs[idx], ys[idx])
        slopes.append(s)
    if not slopes:
        s, _, _ = fit_fn(xs, ys)
        return float(s), float(s)
    lo, hi = np.percentile(slopes, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# decay fits
# ---------------------------------------------------------------------------
def _ols(x, y):
    """Ordinary least squares y = a + m x.  Returns (slope m, intercept a, R^2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    (m, a), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = m * x + a
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0
    return float(m), float(a), float(r2)


def fit_exp(ns, variances):
    """Exponential model log V = a - b n.  Returns (b, a, R^2).

    b ~ log 2 ~ 0.69 signals a barren plateau (V ~ 2^{-n}); b ~ 0 signals no
    exponential decay.
    """
    ns = np.asarray(ns, dtype=float)
    V = np.asarray(variances, dtype=float)
    logV = np.log(np.clip(V, 1e-300, None))
    m, a, r2 = _ols(ns, logV)
    return -m, a, r2       # b = -slope


def fit_poly(ns, variances):
    """Polynomial model log V = a' - c log n (i.e. V ~ n^{-c}).  Returns
    (c, a', R^2).  A good fit with small c and a flat exp-fit => trainable.
    """
    ns = np.asarray(ns, dtype=float)
    V = np.asarray(variances, dtype=float)
    logV = np.log(np.clip(V, 1e-300, None))
    logn = np.log(ns)
    m, a, r2 = _ols(logn, logV)
    return -m, a, r2       # c = -slope


def classify_decay(b_exp, r2_exp, r2_poly):
    """Decision rule from the pre-registered decision rule (report both, let the data speak):

      plateau      if R^2_exp > R^2_poly AND b > 0.3   (clearly exponential)
      trainable    if R^2_poly >= R^2_exp OR b < 0.15
      inconclusive otherwise
    """
    if r2_exp > r2_poly and b_exp > 0.3:
        return "plateau"
    if r2_poly >= r2_exp or b_exp < 0.15:
        return "trainable"
    return "inconclusive"


def crosscheck_fit(variances, dims):
    """Larocca cross-check: log V vs log(1/dim g).  Returns (slope, R^2).

    The Var ~ 1/dim(g) law predicts slope ~ 1 when P_g(rho) P_g(O) is roughly
    n-stable.
    """
    V = np.asarray(variances, dtype=float)
    d = np.asarray(dims, dtype=float)
    logV = np.log(np.clip(V, 1e-300, None))
    log_invdim = np.log(1.0 / np.clip(d, 1e-300, None))
    m, a, r2 = _ols(log_invdim, logV)
    return float(m), float(r2)
