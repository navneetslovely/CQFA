"""
experiments/fisher_information/shift_recovery_cost.py
===================================================
Separate continuous quantum Fisher information from discrete hidden-shift
recovery. A two-sided test of Prop.~\\ref{prop:fisher}.

THE HYPOTHESIS UNDER TEST
-------------------------
prop:fisher (main.tex l.84) currently says, in one sentence, that
  (i)  the Mode-F cross-sector measurement saturates the QCRB (CFI/F_Q = 1.000000), and
  (ii) "a single fixed, post-selection-free product measurement estimates the
       discriminating cross-irrep coherence to additive eps in O(1/eps^2) shots
       INDEPENDENT OF N", and
  (iii) "estimating [the coherence phase] to precision < pi/N recovers log_g x".
Statements (i)-(ii) are LOCAL-ESTIMATION statements while (iii) is a DISCRETE
N-hypothesis statement. Chaining them plausibly costs O(N^2) shots, because eps must
shrink as O(1/N). This module measures that cost rather than asserting it.

SETTING
-------
G = Z_N, U_g|x> = |x+g mod N>.  K = sum_j j |f_j><f_j| (Fourier number operator),
U_theta = exp(-i theta K); U_{2 pi s/N} = shift-by-s.
  |Phi_theta> = (I_R (x) U_theta)|Phi_0>,  |Phi_0> = N^{-1/2} sum_g |g>_R (x) U_g phi.
Half A treats theta in R (continuous).  Half B treats s in {0..N-1} (discrete).

READOUTS COMPARED (all three modes, as required)
------------------------------------------------
  A_projection            Mode A: coherent projection onto the trivial irrep
                          (Pi = |f_0><f_0|), then a fixed system measurement.
                          POST-SELECTED: acceptance = |phi_hat_0|^2 ~ 1/N.
  B_frame_average         Mode B: decohered marginal rho_sys(s), fixed measurement.
  F_srm_joint             Mode F, N-outcome optimal JOINT (square-root / Helstrom)
                          measurement on frame (x) system.  Post-selection-free.
                          This is the readout the IBM Heron C8 run realizes.
  F_fixed_binary_sld      Mode F, the FIXED binary SLD-optimal projective measurement
                          built at theta=0 -- the measurement that attains the QCRB in
                          Half A -- reused verbatim to recover the discrete s.
  F_fixed_product_coh     Mode F, the paper's own single-setting PRODUCT observables
                          O_Re = R_Re (x) M, O_Im = R_Im (x) M, 50/50 randomized,
                          with M chosen to MAXIMISE the coherence (strongest version).
                          This is the estimator sentence (ii) is about.

PRE-REGISTERED CRITERION (two-sided, fixed BEFORE running)
----------------------------------------------------------
Primary quantity: SHOTS_90(N) = the smallest total number of shots at which the
maximum-likelihood decoder over the N shift hypotheses recovers s with probability
>= 0.90 (uniform prior over s), MEASURED by Monte-Carlo, not modelled.
Fit log2(SHOTS_90) = a + b log2(N) over N in {4,8,16,32,64}; report b with a 95% CI.

  * The claim as written SURVIVES  iff, for the readout that sentence (ii) names --
    the single fixed post-selection-free PRODUCT measurement -- the 95% CI for b
    contains 0 (shots flat in N).
  * The claim as written is KILLED iff that CI excludes 0 and is consistent with
    b >= 2, i.e. the discrete recovery cost grows at least like N^2.  In that case
    the restatement is: the COHERENCE is estimable to additive eps in
    O(1/eps^2) shots independent of N at FIXED eps, while RECOVERING THE DISCRETE
    SHIFT from it costs Theta(N^b).
  * MIXED is declared if different Mode-F readouts fall on different sides -- in
    which case the proposition must name which readout it means.
Secondary (does NOT rescue the primary criterion, reported separately): b for
F_srm_joint, the N-outcome joint readout actually run on hardware.

Half A is a separate, independent check: for the CONTINUOUS parameter theta the
QCRB language is legitimate, and CFI/F_Q must reproduce 1.000000, plus an
OPERATIONAL check that the sampled MLE variance meets 1/(M F_Q).

VACUITY GUARD (integrity rule 3)
--------------------------------
For every (readout, N, ensemble) cell we compute
    label_signal_scale = max_{s != s'} TV( p(.|s), p(.|s') )
BEFORE any decoding.  Cells with label_signal_scale < 1e-10 are marked
'vacuous_excluded' and are NOT fitted; for Modes A and B this is exactly the
zero of the separation and is reported as such (with an exact-zero certificate).

CAPS (integrity rule 7)
-----------------------
If SHOTS_90 exceeds the budget cap, the measurement column shots_90 is left EMPTY
and the cap is written to shots_90_lower_bound.

SEEDS (integrity rule 5)
------------------------
>= 5 independent cells per (N, readout): 5 random base states phi for the 'generic'
ensemble; 5 independent RNG replicates for the deterministic 'dlp_flat' ensemble.

Outputs (all NEW files; nothing existing is touched):
  results/fisher_information/shift_recovery_cost.csv          (per-cell measurements)
  results/fisher_information/shift_recovery_cost_fits.csv     (scaling exponents + CI)
  results/fisher_information/shift_recovery_cost_halfA.csv    (continuous-parameter QCRB)
  results/fisher_information/shift_recovery_cost_hw_scaling.csv (IBM N=4,8,16 power check)
"""
from __future__ import annotations

import argparse
import csv
import math
import os

import zlib

import numpy as np

CAP_DEFAULT = 10 ** 8          # shot-budget cap; fixed before the first comparison
TARGET_P = 0.90                # pre-registered success target
TRIALS_SEARCH = 400            # MC trials per bisection probe
TRIALS_CONFIRM = 4000          # MC trials at the returned M*


# ===========================================================================
# Z_N structure (matrix representation: Phi is an (N x N) array A[g, x])
# ===========================================================================
_SHIFT_CACHE: dict = {}


# Deterministic per-readout salt. This replaces abs(hash(name)) % 9973, which was
# PYTHONHASHSEED-salted: the RNG stream differed on every invocation, so the module
# was not reproducible (the fitted exponent moved 2.871 -> 2.888 -> 2.876 across
# three runs and one secondary verdict flipped sign). zlib.crc32 is stable across
# processes and Python versions, and unlike a hardcoded table it cannot silently
# miss a readout name.
def readout_salt(name: str) -> int:
    return zlib.crc32(name.encode("utf-8")) % 9973

def shift(N: int, g: int) -> np.ndarray:
    key = (N, g % N)
    if key not in _SHIFT_CACHE:
        _SHIFT_CACHE[key] = np.roll(np.eye(N, dtype=complex), g % N, axis=0)
    return _SHIFT_CACHE[key]


def fourier_states(N: int) -> np.ndarray:
    """Columns |f_j> = N^{-1/2} sum_x omega^{jx}|x>."""
    x = np.arange(N)
    return np.exp(2j * np.pi * np.outer(x, x) / N) / np.sqrt(N)


def K_operator(N: int) -> np.ndarray:
    Fm = fourier_states(N)
    return (Fm * np.arange(N)) @ Fm.conj().T


def U_theta(N: int, theta: float) -> np.ndarray:
    Fm = fourier_states(N)
    return (Fm * np.exp(-1j * theta * np.arange(N))) @ Fm.conj().T


def rand_state(N: int, rng) -> np.ndarray:
    v = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    return v / np.linalg.norm(v)


def joint_matrix(phi: np.ndarray, N: int, theta: float = 0.0) -> np.ndarray:
    """A[g, x] = (U_g U_theta phi)[x] / sqrt(N);  |Phi_theta> = sum_{g,x} A[g,x] |g>|x>."""
    psi = U_theta(N, theta) @ phi
    A = np.empty((N, N), dtype=complex)
    for g in range(N):
        A[g] = shift(N, g) @ psi
    return A / np.sqrt(N)


def ip(A: np.ndarray, B: np.ndarray) -> complex:
    """<A|B> for matrix-represented joint states."""
    return complex(np.vdot(A, B))


# ===========================================================================
# HALF A -- continuous parameter theta: QFI, QCRB attainment, operational check
# ===========================================================================
def qfi(phi: np.ndarray, N: int):
    """F_Q = 4 Var_{|Phi_0>}(I (x) K) = 4 Var_phi(K).  Returns (F_Q, mean, var)."""
    A = joint_matrix(phi, N)
    K = K_operator(N)
    HA = A @ K.T                                   # (I (x) K)|Phi_0>
    mean = float(np.real(ip(A, HA)))
    mean2 = float(np.real(ip(HA, HA)))
    var = mean2 - mean ** 2
    return 4.0 * var, mean, var


def qfi_analytic(phi: np.ndarray, N: int):
    Fm = fourier_states(N)
    w = np.abs(Fm.conj().T @ phi) ** 2
    w = w / w.sum()
    j = np.arange(N)
    m = float(np.sum(w * j))
    return 4.0 * float(np.sum(w * j ** 2) - m ** 2), w


def sld_basis(phi: np.ndarray, N: int):
    """FIXED (theta-independent) SLD-optimal binary projectors at theta = 0:
    |+/-> = (|Phi_0> +/- i|chi>)/sqrt2, |chi> = (H - <H>)|Phi_0>/DeltaH."""
    A = joint_matrix(phi, N)
    K = K_operator(N)
    HA = A @ K.T
    mean = float(np.real(ip(A, HA)))
    chi = HA - mean * A
    dH = float(np.linalg.norm(chi))
    if dH < 1e-14:
        return None
    chi = chi / dH
    return A, chi, (A + 1j * chi) / np.sqrt(2), (A - 1j * chi) / np.sqrt(2), dH


def cfi_fixed_binary(phi: np.ndarray, N: int, dtheta: float = 1e-5):
    """CFI at theta=0 of the FIXED binary measurement {|+>,|->,rest}."""
    st = sld_basis(phi, N)
    if st is None:
        return 0.0, 0.0
    A0, chi, plus, minus, dH = st

    def probs(th):
        A = joint_matrix(phi, N, th)
        return np.array([abs(ip(plus, A)) ** 2, abs(ip(minus, A)) ** 2])

    p0, pp, pm = probs(0.0), probs(+dtheta), probs(-dtheta)
    dp = (pp - pm) / (2 * dtheta)
    good = p0 > 1e-12
    return float(np.sum(dp[good] ** 2 / p0[good])), 4.0 * dH ** 2


def operational_qcrb(phi: np.ndarray, N: int, F_Q: float, rng,
                     n_rep: int = 4000, shots: int = 400):
    """OPERATIONAL attainment: sample the FIXED binary measurement `shots` times at
    theta_true = 0, take the MLE restricted to the local window |theta| <= 3/sqrt(F_Q)
    (the QCRB is a LOCAL bound), and compare Var(theta_hat) to the QCRB 1/(shots F_Q)."""
    st = sld_basis(phi, N)
    if st is None:
        return float("nan"), float("nan")
    A0, chi, plus, minus, dH = st
    win = 3.0 / math.sqrt(F_Q)
    grid = np.linspace(-win, win, 601)
    pplus = np.array([abs(ip(plus, joint_matrix(phi, N, th))) ** 2 for th in grid])
    pminus = np.array([abs(ip(minus, joint_matrix(phi, N, th))) ** 2 for th in grid])
    prest = np.clip(1.0 - pplus - pminus, 0.0, None)
    P = np.stack([pplus, pminus, prest], axis=1)          # (grid, 3)
    P = np.clip(P, 1e-300, None)
    i0 = int(np.argmin(np.abs(grid)))                      # theta_true = 0
    counts = rng.multinomial(shots, P[i0] / P[i0].sum(), size=n_rep)   # (n_rep, 3)
    ll = counts @ np.log(P).T                              # (n_rep, grid)
    hat = grid[np.argmax(ll, axis=1)]
    var = float(np.var(hat))
    qcrb = 1.0 / (shots * F_Q)
    return var, var / qcrb


# ===========================================================================
# HALF B -- the discrete N-hypothesis problem
# ===========================================================================
def gram_matrix(phi: np.ndarray, N: int) -> np.ndarray:
    """G[s,s'] = <Phi_s|Phi_s'> = <phi|U_{s'-s}|phi> (circulant)."""
    Fm = fourier_states(N)
    w = np.abs(Fm.conj().T @ phi) ** 2
    w = w / w.sum()
    d = np.arange(N)
    c = np.array([np.sum(w * np.exp(-2j * np.pi * np.arange(N) * dd / N)) for dd in d])
    G = np.empty((N, N), dtype=complex)
    for s in range(N):
        for t in range(N):
            G[s, t] = c[(t - s) % N]
    return G


def srm_transition(G: np.ndarray) -> np.ndarray:
    """Square-root (pretty-good) measurement transition probabilities p(j|i) =
    |[G^{1/2}]_{ji}|^2 for a uniform prior.  For a GEOMETRICALLY UNIFORM state set
    (which {|Phi_s>} is, generated by a Z_N unitary) with a uniform prior the SRM is
    the minimum-error (Helstrom) measurement, so this is exact, not a bound."""
    evals, evecs = np.linalg.eigh(G)
    evals = np.clip(evals.real, 0.0, None)
    Gh = (evecs * np.sqrt(evals)) @ evecs.conj().T
    return np.abs(Gh) ** 2


def helstrom_certificate(phi: np.ndarray, N: int, G: np.ndarray):
    """Minimum-error (Helstrom) discrimination of the N hypotheses, with an explicit
    DUAL OPTIMALITY CERTIFICATE rather than an appeal to the geometric-uniformity
    theorem.  Build explicit coordinates c_s with G = C^dag C (so <Phi_s|Phi_t> = G[s,t]),
    form the SRM E_s = rho^{-1/2} (1/N) c_s c_s^dag rho^{-1/2}, rho = (1/N) sum_s c_s c_s^dag,
    and verify the Yuen-Kennedy-Lax conditions
        Y = sum_s (1/N) c_s c_s^dag E_s   Hermitian,   Y - (1/N) c_s c_s^dag >= 0  for all s.
    A non-negative min slack (to numerical tolerance) CERTIFIES the SRM is optimal, so
    P_success is the true minimum-error probability, not merely an achievable value.
    Returns (P_srm, P_closed_form, min_dual_slack, dual_herm_err, povm_completeness_err)."""
    evals, evecs = np.linalg.eigh(G)
    evals = np.clip(evals.real, 0.0, None)
    C = (evecs * np.sqrt(evals)).conj().T                  # C[:, s] = coords of |Phi_s>
    rho = (C @ C.conj().T) / N
    rv, rU = np.linalg.eigh(rho)
    inv_sqrt = np.where(rv > 1e-13, 1.0 / np.sqrt(np.clip(rv, 1e-300, None)), 0.0)
    rho_mh = (rU * inv_sqrt) @ rU.conj().T                 # rho^{-1/2} on the support

    Es, P_srm = [], 0.0
    for s in range(N):
        cs = C[:, s]
        E = rho_mh @ (np.outer(cs, cs.conj()) / N) @ rho_mh
        Es.append(E)
        P_srm += float(np.real(cs.conj() @ E @ cs)) / N
    completeness = float(np.max(np.abs(sum(Es) - rU @ np.diag((rv > 1e-13).astype(float))
                                        @ rU.conj().T)))

    Y = sum((np.outer(C[:, s], C[:, s].conj()) / N) @ Es[s] for s in range(N))
    herm_err = float(np.max(np.abs(Y - Y.conj().T)))
    Yh = (Y + Y.conj().T) / 2
    min_slack = np.inf
    for s in range(N):
        cs = C[:, s]
        slack = Yh - np.outer(cs, cs.conj()) / N
        min_slack = min(min_slack, float(np.linalg.eigvalsh((slack + slack.conj().T) / 2).min()))

    Fm = fourier_states(N)
    w = np.abs(Fm.conj().T @ phi) ** 2
    w = w / w.sum()
    P_closed = float((np.sum(np.sqrt(w)) ** 2) / N)
    return P_srm, P_closed, min_slack, herm_err, completeness


# ---------------------------------------------------------------------------
# outcome distributions p(o | s) for each readout  (all measurements FIXED in s)
# ---------------------------------------------------------------------------
def readout_A_projection(phi: np.ndarray, N: int):
    """Mode A: post-select on Pi = |f_0><f_0|, then measure the system in the
    computational basis.  Accepted state is |f_0> for every s -> uniform, s-blind.
    Returns (P[o,s], acceptance)."""
    Fm = fourier_states(N)
    f0 = Fm[:, 0]
    accs, cols = [], []
    for s in range(N):
        psi = shift(N, s) @ phi
        amp = complex(np.vdot(f0, psi))
        accs.append(abs(amp) ** 2)
        post = f0                                          # |f_0><f_0|psi normalised
        cols.append(np.abs(post) ** 2)
    P = np.stack(cols, axis=1)
    P = P / P.sum(0, keepdims=True)
    return P, float(np.mean(accs))


def readout_B_frame_average(phi: np.ndarray, N: int):
    """Mode B: rho_sys(s) = (1/N) sum_g U_g psi_s psi_s^dag U_g^dag, computational basis."""
    cols = []
    for s in range(N):
        psi = shift(N, s) @ phi
        rho = np.zeros((N, N), dtype=complex)
        for g in range(N):
            v = shift(N, g) @ psi
            rho += np.outer(v, v.conj())
        rho /= N
        cols.append(np.clip(np.real(np.diag(rho)), 0, None))
    P = np.stack(cols, axis=1)
    P = P / P.sum(0, keepdims=True)
    return P, 1.0


def readout_F_srm(phi: np.ndarray, N: int, G: np.ndarray):
    """Mode F: the N-outcome optimal JOINT (square-root) measurement.  p(o|s)."""
    T = srm_transition(G)                                  # T[j, i] = p(j | i)
    P = T / T.sum(0, keepdims=True)
    return P, 1.0


def readout_F_fixed_binary(phi: np.ndarray, N: int):
    """Mode F: the FIXED binary SLD-optimal measurement from Half A, {+, -, rest}."""
    st = sld_basis(phi, N)
    if st is None:
        return None, 1.0
    A0, chi, plus, minus, dH = st
    cols = []
    for s in range(N):
        A = joint_matrix(phi, N, 2 * np.pi * s / N)
        pp = abs(ip(plus, A)) ** 2
        pm = abs(ip(minus, A)) ** 2
        cols.append([pp, pm, max(0.0, 1.0 - pp - pm)])
    P = np.array(cols).T
    P = P / P.sum(0, keepdims=True)
    return P, 1.0


def readout_F_fixed_product(phi: np.ndarray, N: int, kpair=(0, 1), strongest=True,
                            rng=None):
    """Mode F: the paper's SINGLE-SETTING POST-SELECTION-FREE PRODUCT observables.
    Frame is measured in the fixed basis diagonalising R_Re (resp. R_Im) built from the
    Fourier frame vectors |e_k>, |e_k'>; the system is measured in the eigenbasis of a
    FIXED Hermitian M with ||M|| = 1.  Settings are randomised 50/50, so the protocol is
    a single fixed post-selection-free product measurement per shot.
    strongest=True picks the M that MAXIMISES |v_k^dag M v_k'| (integrity rule 4:
    report against the strongest available version of our own estimator)."""
    k, kp = kpair
    Fm_frame = fourier_states(N)                          # |e_a> columns
    ek, ekp = Fm_frame[:, k], Fm_frame[:, kp]

    # frame bases for the Re and Im settings
    bRe = [(ek + ekp) / np.sqrt(2), (ek - ekp) / np.sqrt(2)]
    bIm = [(ek + 1j * ekp) / np.sqrt(2), (ek - 1j * ekp) / np.sqrt(2)]
    rest = [Fm_frame[:, a] for a in range(N) if a not in (k, kp)]
    BRe = np.stack(bRe + rest, axis=1)
    BIm = np.stack(bIm + rest, axis=1)

    # v_k at s = 0 fixes the optimal system observable M (independent of s)
    def vcomp(kk, s):
        psi = shift(N, s) @ phi
        acc = np.zeros(N, dtype=complex)
        for g in range(N):
            acc += np.exp(-2j * np.pi * g * kk / N) * (shift(N, g) @ psi)
        return acc / N

    v0k, v0kp = vcomp(k, 0), vcomp(kp, 0)
    if strongest and np.linalg.norm(v0k) > 1e-12 and np.linalg.norm(v0kp) > 1e-12:
        uk = v0k / np.linalg.norm(v0k)
        ukp = v0kp / np.linalg.norm(v0kp)
        M = np.outer(uk, ukp.conj()) + np.outer(ukp, uk.conj())
    else:
        H = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        M = (H + H.conj().T) / 2
    M = M / np.linalg.norm(M, 2)                          # ||M||_op = 1
    _, Msys = np.linalg.eigh(M)

    cols = []
    coh_abs = None
    for s in range(N):
        A = joint_matrix(phi, N, 2 * np.pi * s / N)       # A[g, x]
        # amplitudes in (frame basis) x (system eigenbasis of M)
        ARe = BRe.conj().T @ A @ Msys.conj()
        AIm = BIm.conj().T @ A @ Msys.conj()
        pRe = (np.abs(ARe) ** 2).ravel()
        pIm = (np.abs(AIm) ** 2).ravel()
        cols.append(0.5 * np.concatenate([pRe, pIm]))
        if s == 0:
            vk, vkp = vcomp(k, 0), vcomp(kp, 0)
            coh_abs = float(abs(vk.conj() @ M @ vkp))
    P = np.stack(cols, axis=1)
    P = P / P.sum(0, keepdims=True)
    return P, 1.0, coh_abs


# ---------------------------------------------------------------------------
# vacuity guard + empirical shots-to-90% (measured, not modelled)
# ---------------------------------------------------------------------------
def label_signal_scale(P: np.ndarray) -> float:
    """max_{s != s'} TV(p(.|s), p(.|s'))."""
    N = P.shape[1]
    best = 0.0
    for a in range(N):
        d = 0.5 * np.abs(P[:, [a]] - P).sum(0)
        d[a] = 0.0
        best = max(best, float(d.max()))
    return best


def _success_rate(P: np.ndarray, M: int, trials: int, rng) -> float:
    """MC success probability of the ML decoder over N hypotheses at M shots.
    Ties (all log-likelihoods equal, e.g. a blind readout) are broken uniformly."""
    n_out, N = P.shape
    logP = np.log(np.clip(P, 1e-300, None))
    per = max(1, trials // N)
    ok = 0
    tot = 0
    for s in range(N):
        counts = rng.multinomial(M, P[:, s] / P[:, s].sum(), size=per)
        ll = counts @ logP                                # (per, N)
        mx = ll.max(1, keepdims=True)
        tied = (ll >= mx - 1e-9)
        ntie = tied.sum(1)
        if np.all(ntie == 1):
            pick = np.argmax(ll, axis=1)
        else:
            # break ties uniformly at random (blind readouts tie across all N)
            u = rng.random(tied.shape) * tied
            pick = np.argmax(u, axis=1)
        ok += int(np.sum(pick == s))
        tot += per
    return ok / tot


def shots_to_target(P: np.ndarray, rng, cap: int = CAP_DEFAULT,
                    target: float = TARGET_P):
    """Smallest M with P(success) >= target, by exponential search + integer bisection.
    Returns (M_star or None if censored, confirmed success rate at M_star)."""
    if _success_rate(P, cap, TRIALS_SEARCH, rng) < target:
        return None, _success_rate(P, cap, TRIALS_CONFIRM, rng)
    hi = 1
    while hi <= cap and _success_rate(P, hi, TRIALS_SEARCH, rng) < target:
        hi *= 2
    hi = min(hi, cap)
    lo = max(1, hi // 2)
    while lo < hi:
        mid = (lo + hi) // 2
        if _success_rate(P, mid, TRIALS_SEARCH, rng) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo, _success_rate(P, lo, TRIALS_CONFIRM, rng)


# ===========================================================================
# fits
# ===========================================================================
def loglog_fit(Ns, ys, sds):
    """OLS of log2(y) on log2(N); returns (slope, se, lo95, hi95, n)."""
    x = np.log2(np.asarray(Ns, dtype=float))
    y = np.log2(np.asarray(ys, dtype=float))
    n = len(x)
    if n < 3:
        return (float("nan"),) * 4 + (n,)
    X = np.stack([np.ones(n), x], axis=1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - 2
    s2 = float(resid @ resid) / dof if dof > 0 else float("nan")
    cov = s2 * np.linalg.inv(X.T @ X)
    se = math.sqrt(max(cov[1, 1], 0.0))
    try:
        from scipy import stats
        t = float(stats.t.ppf(0.975, dof))
    except Exception:
        t = 12.71 if dof == 1 else (4.303 if dof == 2 else 2.776)
    return float(beta[1]), se, float(beta[1] - t * se), float(beta[1] + t * se), n


# ===========================================================================
# Can the N = 4, 8, 16 hardware run support an ASYMPTOTIC Theta(N^2) claim?
# (No QPU jobs; this reads results/ibm_campaign/* already on disk.)
# ===========================================================================
HW_FILES = ["results/ibm_campaign/ibm_modef/analysis/c8.csv",
            "results/ibm_campaign/ibm_modef2/analysis/c8.csv"]


def hardware_scaling_check(out_dir):
    rows = []
    obs = {}
    for path in HW_FILES:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                if r.get("mode") != "F":
                    continue
                N = int(r["N"])
                obs.setdefault(N, {"peak": [], "tv": [], "err": []})
                obs[N]["peak"].append(float(r["peak_prob"]))
                obs[N]["tv"].append(float(r["tv_from_uniform"]))
                obs[N]["err"].append(float(r["abs_err"]))
    Ns = sorted(obs)
    for quantity in ["peak", "tv", "err"]:
        ys = [float(np.mean(obs[N][quantity])) for N in Ns]
        sd = [float(np.std(obs[N][quantity])) for N in Ns]
        b, se, lo, hi, n = loglog_fit(Ns, ys, sd)
        rows.append(dict(
            source=";".join(HW_FILES), quantity=quantity,
            Ns=";".join(map(str, Ns)), n_distinct_N=len(Ns),
            means=";".join(f"{v:.6g}" for v in ys),
            stds=";".join(f"{v:.6g}" for v in sd),
            fitted_slope_b=b, slope_se=se, ci95_lo=lo, ci95_hi=hi,
            ci95_width=(hi - lo), dof=len(Ns) - 2,
            measures_fisher_information=0,
            note=("hardware observable is a RECOVERY PROBABILITY / distance, not a Fisher "
                  "information; its ideal value is N-independent (ideal_peak_prob=1 at "
                  "N=4,8,16), so it carries no N^2 content by construction"),
        ))
    # how wide is a 3-point power-law CI, structurally?
    try:
        from scipy import stats
        t1 = float(stats.t.ppf(0.975, 1))
    except Exception:
        t1 = 12.706
    rows.append(dict(
        source="structural", quantity="power_law_identifiability",
        Ns=";".join(map(str, Ns)), n_distinct_N=len(Ns), means="", stds="",
        fitted_slope_b="", slope_se="", ci95_lo="", ci95_hi="", ci95_width="",
        dof=len(Ns) - 2, measures_fisher_information=0,
        note=(f"with {len(Ns)} distinct N the power-law fit has {len(Ns)-2} residual dof; "
              f"t_(0.975,{len(Ns)-2}) = {t1:.3f}, so the 95% CI on the exponent is "
              f"+/-{t1:.2f} residual standard errors -- an exponent of 1, 1.5, 2 or 3 "
              f"cannot be separated from N=4,8,16 at any realistic shot noise"),
    ))
    out = os.path.join(out_dir, "shift_recovery_cost_hw_scaling.csv")
    with open(out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print()
    print("=" * 78)
    print("IBM N=4,8,16 -> can it support an ASYMPTOTIC Theta(N^2) Fisher claim?")
    print("=" * 78)
    for r in rows[:-1]:
        print(f"  {r['quantity']:<6} N={r['Ns']}  b={r['fitted_slope_b']:+.3f} "
              f"95%CI=[{r['ci95_lo']:+.3f},{r['ci95_hi']:+.3f}] width={r['ci95_width']:.2f}")
    print(f"  {len(Ns)} distinct N, dof={len(Ns)-2}, t(0.975,{len(Ns)-2})={t1:.3f}")
    print("  -> NO Fisher information was measured on hardware at any N.")
    print("wrote", out)
    return rows


# ===========================================================================
# driver
# ===========================================================================
def run(sizes, seeds, cap, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rowsA, rowsB = [], []

    # ---------------- HALF A ----------------
    print("=" * 78)
    print("HALF A  --  CONTINUOUS parameter theta:  QFI, QCRB attainment (legitimate regime)")
    print("=" * 78)
    for N in sizes:
        for sd in seeds:
            rng = np.random.default_rng(100 + sd)          # same stream as lc_qcrb_certificate
            phi = rand_state(N, rng)
            FQ, meanH, varH = qfi(phi, N)
            FQa, w = qfi_analytic(phi, N)
            cfi_fd, cfi_an = cfi_fixed_binary(phi, N)
            ratio = cfi_fd / FQ if FQ > 1e-12 else float("nan")
            vr = np.random.default_rng(9000 + 17 * N + sd)
            var_hat, var_over_qcrb = operational_qcrb(phi, N, FQ, vr)
            rowsA.append(dict(
                half="A_continuous", N=N, seed=sd,
                F_Q_numeric=FQ, F_Q_analytic=FQa, F_Q_abs_gap=abs(FQ - FQa),
                var_K=varH, F_Q_over_N2=FQ / N ** 2,
                CFI_fixed_binary=cfi_fd, CFI_over_F_Q=ratio,
                mle_shots=400, mle_var_theta=var_hat, qcrb_var=1.0 / (400 * FQ),
                mle_var_over_qcrb=var_over_qcrb,
                label_signal_scale=FQ,
                vacuity_flag="ok" if FQ > 1e-10 else "vacuous_excluded",
            ))
        sub = [r for r in rowsA if r["N"] == N]
        print(f"  N={N:>3}  F_Q={np.mean([r['F_Q_numeric'] for r in sub]):9.2f}  "
              f"CFI/F_Q={np.mean([r['CFI_over_F_Q'] for r in sub]):.6f}  "
              f"MLE Var/QCRB={np.mean([r['mle_var_over_qcrb'] for r in sub]):.3f}")

    # ---------------- HALF B ----------------
    print()
    print("=" * 78)
    print("HALF B  --  DISCRETE hidden shift s in {0..N-1}:  N-hypothesis discrimination")
    print("=" * 78)
    ensembles = [("dlp_flat", None)] + [("generic", sd) for sd in seeds]

    for N in sizes:
        for ens, sd in ensembles:
            if ens == "dlp_flat":
                phi = np.zeros(N, dtype=complex)
                phi[0] = 1.0                                # uniform Fourier weights
                reps = list(range(5))                       # 5 RNG replicates
            else:
                phi = rand_state(N, np.random.default_rng(100 + sd))
                reps = [0]

            G = gram_matrix(phi, N)
            offdiag = float(np.max(np.abs(G - np.diag(np.diag(G)))))
            P_srm, P_closed, min_slack, herm_err, comp_err = helstrom_certificate(phi, N, G)
            orthogonal = offdiag < 1e-12
            FQ, _, _ = qfi(phi, N)

            # exact-zero certificate for the blind marginal (the "0" of the separation)
            def _rho_sys(ss):
                psi = shift(N, ss) @ phi
                r = np.zeros((N, N), dtype=complex)
                for g in range(N):
                    v = shift(N, g) @ psi
                    r += np.outer(v, v.conj())
                return r / N
            r0 = _rho_sys(0)
            blind_dev = max(float(np.max(np.abs(_rho_sys(ss) - r0))) for ss in range(N))

            builds = {}
            PA, aA = readout_A_projection(phi, N)
            builds["A_projection"] = (PA, aA, "", 1)
            PB, aB = readout_B_frame_average(phi, N)
            builds["B_frame_average"] = (PB, aB, "", 1)
            PF, aF = readout_F_srm(phi, N, G)
            builds["F_srm_joint"] = (PF, aF, "", 1)
            PS, aS = readout_F_fixed_binary(phi, N)
            if PS is not None:
                builds["F_fixed_binary_sld"] = (PS, aS, "", 1)
            prng = np.random.default_rng(5000 + N + (sd or 0))
            PP, aP, coh = readout_F_fixed_product(phi, N, (0, 1), True, prng)
            builds["F_fixed_product_coh_bestM"] = (PP, aP, coh, 2)
            prng2 = np.random.default_rng(7000 + N + (sd or 0))
            PR, aR, cohR = readout_F_fixed_product(phi, N, (0, 1), False, prng2)
            builds["F_fixed_product_coh_randM"] = (PR, aR, cohR, 2)

            for name, (P, acc, coh_abs, nset) in builds.items():
                lss = label_signal_scale(P)
                vac = "vacuous_excluded" if lss < 1e-10 else "ok"
                for rp in reps:
                    # NOTE: this previously used abs(hash(name)) % 9973. Python salts
                    # str.__hash__ per process (PYTHONHASHSEED), so the RNG stream
                    # differed on every invocation and the module was NOT reproducible:
                    # the fitted exponent moved 2.871 -> 2.888 -> 2.876 across three
                    # runs and one secondary verdict flipped sign. READOUT_SALT is a
                    # fixed table so the stream depends only on (N, seed, rep, readout).
                    rng = np.random.default_rng(
                        int(1_000_003 * N + 7919 * (sd if sd is not None else 99)
                            + 131 * rp + readout_salt(name)))
                    if vac == "vacuous_excluded":
                        Mstar = None
                        conf = _success_rate(P, cap, TRIALS_CONFIRM, rng)
                    else:
                        Mstar, conf = shots_to_target(P, rng, cap, TARGET_P)
                    rowsB.append(dict(
                        half="B_discrete", N=N, ensemble=ens,
                        seed=(sd if sd is not None else ""), replicate=rp,
                        readout=name,
                        n_hypotheses=N, n_fixed_settings=nset,
                        gram_offdiag_max=offdiag,
                        hypotheses_mutually_orthogonal=int(orthogonal),
                        P_exact_when_orthogonal=(1.0 if orthogonal else ""),
                        P_success_helstrom_srm=P_srm,
                        P_success_closed_form=P_closed,
                        helstrom_srm_gap=abs(P_srm - P_closed),
                        helstrom_dual_min_slack=min_slack,
                        helstrom_dual_herm_err=herm_err,
                        helstrom_povm_completeness_err=comp_err,
                        bayes_risk_uniform_prior=1.0 - P_srm,
                        F_Q=FQ,
                        blind_marginal_max_dev=blind_dev,
                        label_signal_scale=lss,
                        vacuity_flag=vac,
                        # --- the local-estimation and discrete-recovery costs ---
                        postselection_acceptance=acc,
                        postselection_overhead_factor=(1.0 / acc if acc > 0 else ""),
                        coherence_abs_at_s0=(coh_abs if coh_abs != "" else ""),
                        shots_90=("" if Mstar is None else Mstar),
                        shots_90_lower_bound=("" if Mstar is not None else cap),
                        shots_90_incl_postselection=("" if Mstar is None
                                                     else (Mstar / acc if acc > 0 else "")),
                        shots_90_over_N2=("" if Mstar is None else Mstar / N ** 2),
                        success_rate_measured=conf,
                        success_rate_measured_at_shots=(cap if Mstar is None else Mstar),
                        target_success=TARGET_P,
                        shot_cap=cap,
                    ))
            done = [r for r in rowsB if r["N"] == N and r["ensemble"] == ens
                    and (r["seed"] == (sd if sd is not None else ""))]
            msg = "  ".join(
                f"{r['readout'].split('_')[0]}{'/' + r['readout'].split('_')[1] if r['readout'].count('_') else ''}"
                f"={'>' + str(r['shots_90_lower_bound']) if r['shots_90'] == '' else r['shots_90']}"
                for r in done if r["replicate"] == 0)
            print(f"  N={N:>3} {ens:<8} seed={str(sd):<4} orth={int(orthogonal)} "
                  f"P_Hel={P_srm:.4f}  {msg}")

    # ---------------- write per-cell CSVs ----------------
    outB = os.path.join(out_dir, "shift_recovery_cost.csv")
    with open(outB, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rowsB[0].keys()))
        wtr.writeheader()
        wtr.writerows(rowsB)
    outA = os.path.join(out_dir, "shift_recovery_cost_halfA.csv")
    with open(outA, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rowsA[0].keys()))
        wtr.writeheader()
        wtr.writerows(rowsA)

    # ---------------- fits ----------------
    print()
    print("=" * 78)
    print("SCALING OF THE MEASURED SHOT COUNT:  log2(shots_90) = a + b log2(N)")
    print("=" * 78)
    fits = []
    readouts = sorted({r["readout"] for r in rowsB})
    for ens in ["dlp_flat", "generic"]:
        for name in readouts:
            Ns, ys, sds_, censored, vac = [], [], [], 0, 0
            for N in sizes:
                cells = [r for r in rowsB if r["readout"] == name and r["N"] == N
                         and r["ensemble"] == ens]
                if not cells:
                    continue
                if all(c["vacuity_flag"] == "vacuous_excluded" for c in cells):
                    vac += 1
                    continue
                vals = [c["shots_90"] for c in cells if c["shots_90"] != ""]
                if len(vals) < len(cells):
                    censored += 1
                if not vals:
                    continue
                Ns.append(N)
                ys.append(float(np.mean(vals)))
                sds_.append(float(np.std(vals)))
            if len(Ns) >= 3:
                b, se, lo, hi, n = loglog_fit(Ns, ys, sds_)
            else:
                b = se = lo = hi = float("nan"); n = len(Ns)
            # explicit, non-collapsing verdict: state what the CI excludes.
            exc0 = cont2 = exc2hi = ""
            verdict = "no_fit"
            if not math.isnan(lo):
                exc0 = int(not (lo <= 0.0 <= hi))
                cont2 = int(lo <= 2.0 <= hi)
                exc2hi = int(lo > 2.0)
                parts = [f"b={b:+.2f}"]
                parts.append("CI EXCLUDES 0 (grows with N)" if exc0
                             else "CI contains 0 (cannot exclude flat)")
                if exc2hi:
                    parts.append("CI EXCLUDES 2 from above (grows FASTER than N^2)")
                elif cont2:
                    parts.append("CI contains 2 (consistent with N^2)")
                elif hi < 2.0:
                    parts.append("CI excludes 2 from below (slower than N^2)")
                verdict = "; ".join(parts)
            elif vac and not Ns:
                verdict = "blind: p(o|s) exactly independent of s at every N (vacuous_excluded)"
            elif censored and not Ns:
                verdict = "censored at every N: never reaches 0.90 within the shot cap"
            fits.append(dict(
                ensemble=ens, readout=name, n_points=n,
                Ns=";".join(str(x) for x in Ns),
                shots_90_mean=";".join(f"{v:.6g}" for v in ys),
                shots_90_std=";".join(f"{v:.6g}" for v in sds_),
                slope_b=b, slope_se=se, ci95_lo=lo, ci95_hi=hi,
                ci95_excludes_0=exc0, ci95_contains_2=cont2, ci95_excludes_2_above=exc2hi,
                n_N_censored=censored, n_N_vacuous=vac, verdict=verdict,
            ))
            print(f"  {ens:<9} {name:<26} b={b:+.3f} +/- {se:.3f}  "
                  f"95%CI=[{lo:+.3f},{hi:+.3f}]  {verdict}"
                  + (f"   [{vac} N vacuous/blind]" if vac else "")
                  + (f"   [{censored} N censored]" if censored else ""))
    outF = os.path.join(out_dir, "shift_recovery_cost_fits.csv")
    with open(outF, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(fits[0].keys()))
        wtr.writeheader()
        wtr.writerows(fits)

    hardware_scaling_check(out_dir)

    print()
    print("wrote", outB)
    print("wrote", outA)
    print("wrote", outF)
    return rowsA, rowsB, fits


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--cap", type=int, default=CAP_DEFAULT)
    ap.add_argument("--out-dir", default="results/fisher_information")
    a = ap.parse_args(argv)
    run(a.sizes, a.seeds, a.cap, a.out_dir)


if __name__ == "__main__":
    main()
