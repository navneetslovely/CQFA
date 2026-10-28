
from __future__ import annotations

import csv
import itertools
import os
import sys

import numpy as np
from scipy.optimize import minimize

_ROOT = os.path.abspath(__file__)
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(
        os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
ROOT = os.path.abspath(__file__)
while ROOT != os.path.dirname(ROOT) and not os.path.exists(
        os.path.join(ROOT, "pyproject.toml")):
    ROOT = os.path.dirname(ROOT)
OUT = os.path.join(ROOT, "results", "cqfa_pp")

N_SEEDS = 6
SEEDS = list(range(N_SEEDS))
N_TEST = 2000
SIGMA_GRID = [0.5, 1.0, 2.0, 4.0]
NTRAIN_GRID = [40, 80, 160]
PRIMARY_CELL = (2.0, 80)                 # pre-registered primary cell
L2_GRID = [1e-3, 1e-2, 1e-1, 1.0]        # selected per arm/seed on a train-internal val split
VAL_FRAC = 0.25
N_RESTARTS = 5                           # 1 uniform init + 4 random inits (never init at H)
MAXITER = 500
N_RANDOM_FRAMES = 200
TOL_ORACLE = 0.03                        # criterion (b)
VACUITY_TOL = 1e-10                      # criterion: label_signal_scale floor


# ===========================================================================
# S4 and its regular representation
# ===========================================================================
def perm_mul(p, q):
    """(p*q)[i] = p[q[i]]."""
    return tuple(p[q[i]] for i in range(len(q)))


def s4_elements():
    """All 24 permutations of {0,1,2,3}, identity first."""
    els = [tuple(p) for p in itertools.permutations(range(4))]
    ident = tuple(range(4))
    els.remove(ident)
    return [ident] + sorted(els)


ELEMENTS = s4_elements()
INDEX = {g: i for i, g in enumerate(ELEMENTS)}
ORDER = len(ELEMENTS)


def regular_rep():
    """Left regular representation: U_g |h> = |g*h>, real permutation matrices (24x24)."""
    U = np.zeros((ORDER, ORDER, ORDER))
    for gi, g in enumerate(ELEMENTS):
        for hi, h in enumerate(ELEMENTS):
            U[gi, INDEX[perm_mul(g, h)], hi] = 1.0
    return U


U_REG = regular_rep()


def subgroup_closure(gens):
    """Closure of `gens` under multiplication, as a frozenset of permutation tuples."""
    ident = tuple(range(4))
    elems = {ident}
    frontier = [ident]
    while frontier:
        a = frontier.pop()
        for g in gens:
            b = perm_mul(a, g)
            if b not in elems:
                elems.add(b)
                frontier.append(b)
    return frozenset(elems)


# (name, generators)  -- true subgroup and its ORDER-MATCHED wrong partner
CYC3_A = subgroup_closure([(1, 2, 0, 3)])                     # <(012)>, order 3
CYC3_B = subgroup_closure([(1, 3, 2, 0)])                     # <(013)>, order 3
KLEIN_V = subgroup_closure([(1, 0, 3, 2), (2, 3, 0, 1)])      # normal Klein V4, order 4
CYC4 = subgroup_closure([(1, 2, 3, 0)])                       # <(0123)>, order 4
CYC2_A = subgroup_closure([(1, 0, 2, 3)])                     # <(01)>, order 2
CYC2_B = subgroup_closure([(1, 0, 3, 2)])                     # <(01)(23)>, order 2
S3_STAB3 = subgroup_closure([(1, 0, 2, 3), (1, 2, 0, 3)])     # stabiliser of 3, order 6
S3_STAB0 = subgroup_closure([(0, 2, 1, 3), (0, 2, 3, 1)])     # stabiliser of 0, order 6

SUBGROUP_PAIRS = [
    ("C2", CYC2_A, "C2_other", CYC2_B),
    ("C3", CYC3_A, "C3_other", CYC3_B),
    ("V4", KLEIN_V, "C4", CYC4),
    ("C4", CYC4, "V4", KLEIN_V),
    ("S3", S3_STAB3, "S3_other", S3_STAB0),
]


def uniform_p_on(H):
    p = np.zeros(ORDER)
    for h in H:
        p[INDEX[h]] = 1.0 / len(H)
    return p


def operator(p):
    """A(p) = sum_g p_g U_g."""
    return np.tensordot(p, U_REG, axes=(0, 0))


# ===========================================================================
# task
# ===========================================================================
def make_data(H, sigma, n, rng):
    """x = normalise(t_y + sigma * (I - Pi_H) eta); templates t_y are H-invariant."""
    Pi = operator(uniform_p_on(H))
    Q = np.eye(ORDER) - Pi
    y = rng.integers(0, 2, size=n)
    eta = rng.standard_normal((n, ORDER))
    noise = eta @ Q.T
    X = TEMPLATES[y] + sigma * noise
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, y.astype(float), Pi


# ===========================================================================
# model: logistic head on A(p) x, optional trainable p
# ===========================================================================
def _loss_grad_head(w, b, Z, y, l2):
    z = Z @ w + b
    # numerically stable logistic loss
    ll = np.logaddexp(0.0, z) - y * z
    loss = ll.mean() + l2 * w @ w
    r = (1.0 / (1.0 + np.exp(-z)) - y) / len(y)
    return loss, r


def fit_head(Z, y, l2):
    """Convex logistic regression on fixed features Z. Global optimum."""
    d = Z.shape[1]

    def obj(v):
        w, b = v[:d], v[d]
        loss, r = _loss_grad_head(w, b, Z, y, l2)
        return loss, np.concatenate([Z.T @ r + 2 * l2 * w, [r.sum()]])

    res = minimize(obj, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                   options={"maxiter": MAXITER})
    return res.x[:d], res.x[d]


def predict(Z, w, b):
    return (Z @ w + b > 0).astype(float)


def fit_joint(X, y, l2, theta0):
    """Joint optimisation over (theta -> p = theta^2/||theta||^2, w, b). Analytic gradient."""
    d = ORDER

    def obj(v):
        theta, w, b = v[:d], v[d:2 * d], v[2 * d]
        S = theta @ theta
        p = theta * theta / S
        A = operator(p)
        Z = X @ A.T
        loss, r = _loss_grad_head(w, b, Z, y, l2)
        dA = np.outer(w, X.T @ r)                       # dL/dA
        dp = np.tensordot(U_REG, dA, axes=([1, 2], [0, 1]))
        dtheta = (2.0 * theta / S) * (dp - dp @ p)
        return loss, np.concatenate([dtheta, Z.T @ r + 2 * l2 * w, [r.sum()]])

    res = minimize(obj, theta0, jac=True, method="L-BFGS-B", options={"maxiter": MAXITER})
    theta = res.x[:d]
    p = theta * theta / (theta @ theta)
    return p, res.x[d:2 * d], res.x[2 * d], res.fun


def select_l2_fixed(Xtr, ytr, A, rng):
    """Pick L2 on a train-internal validation split (never touches test)."""
    n = len(ytr)
    idx = rng.permutation(n)
    nval = max(4, int(VAL_FRAC * n))
    vi, fi = idx[:nval], idx[nval:]
    Z = Xtr @ A.T
    best, best_l2 = -1.0, L2_GRID[0]
    for l2 in L2_GRID:
        w, b = fit_head(Z[fi], ytr[fi], l2)
        acc = (predict(Z[vi], w, b) == ytr[vi]).mean()
        if acc > best:
            best, best_l2 = acc, l2
    return best_l2


def run_fixed_arm(Xtr, ytr, Xte, yte, p, rng):
    A = operator(p)
    l2 = select_l2_fixed(Xtr, ytr, A, rng)
    w, b = fit_head(Xtr @ A.T, ytr, l2)
    return (predict(Xte @ A.T, w, b) == yte).mean(), l2


def run_discovered_arm(Xtr, ytr, Xte, yte, rng):
    """Trainable frame. L2 by train-internal validation; restart chosen by TRAIN loss."""
    n = len(ytr)
    idx = rng.permutation(n)
    nval = max(4, int(VAL_FRAC * n))
    vi, fi = idx[:nval], idx[nval:]

    def best_over_restarts(Xf, yf, l2, seed_rng):
        inits = [np.ones(ORDER)]                                    # uniform over G_ambient
        for _ in range(N_RESTARTS - 1):
            inits.append(np.abs(seed_rng.standard_normal(ORDER)) + 1e-3)
        best = None
        for t0 in inits:
            v0 = np.concatenate([t0, np.zeros(ORDER + 1)])
            p, w, b, f = fit_joint(Xf, yf, l2, v0)
            if best is None or f < best[3]:
                best = (p, w, b, f)
        return best

    best_acc, best_l2 = -1.0, L2_GRID[0]
    for l2 in L2_GRID:
        p, w, b, _ = best_over_restarts(Xtr[fi], ytr[fi], l2, np.random.default_rng(rng.integers(1 << 30)))
        acc = (predict(Xtr[vi] @ operator(p).T, w, b) == ytr[vi]).mean()
        if acc > best_acc:
            best_acc, best_l2 = acc, l2

    p, w, b, _ = best_over_restarts(Xtr, ytr, best_l2, np.random.default_rng(rng.integers(1 << 30)))
    acc = (predict(Xte @ operator(p).T, w, b) == yte).mean()
    return acc, p, best_l2


# ===========================================================================
# structural checks + random-frame baseline
# ===========================================================================
def structure_checks():
    rows = []
    ident = np.eye(ORDER)
    err_closed = max(
        float(np.abs(U_REG[INDEX[perm_mul(g, h)]] - U_REG[INDEX[g]] @ U_REG[INDEX[h]]).max())
        for g in ELEMENTS[:6] for h in ELEMENTS[:6]
    )
    rows.append(("regular_rep_homomorphism_err", err_closed))
    rows.append(("identity_is_first", float(np.abs(U_REG[0] - ident).max())))
    for name, H, _, _ in SUBGROUP_PAIRS:
        Pi = operator(uniform_p_on(H))
        rows.append((f"Pi_{name}_idempotent_err", float(np.abs(Pi @ Pi - Pi).max())))
        rows.append((f"Pi_{name}_rank", float(np.linalg.matrix_rank(Pi, tol=1e-9))))
        rows.append((f"Pi_{name}_expected_rank", float(ORDER / len(H))))
    return rows


def random_frame_mass(H, rng, n=N_RANDOM_FRAMES):
    """mass_H under c ~ CN(0, I); expectation is |H|/|G_ambient|."""
    c = rng.standard_normal((n, ORDER)) + 1j * rng.standard_normal((n, ORDER))
    p = np.abs(c) ** 2
    p /= p.sum(axis=1, keepdims=True)
    hidx = [INDEX[h] for h in H]
    return p[:, hidx].sum(axis=1)


# ===========================================================================
# main
# ===========================================================================
TEMPLATES = None   # set per (subgroup, seed)


def main():
    global TEMPLATES
    os.makedirs(OUT, exist_ok=True)

    with open(os.path.join(OUT, "u2_structure_check.csv"), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["quantity", "value"])
        for k, v in structure_checks():
            wtr.writerow([k, repr(v)])

    rows = []
    for true_name, H, wrong_name, Hw in SUBGROUP_PAIRS:
        Pi_true = operator(uniform_p_on(H))
        hidx = [INDEX[h] for h in H]
        for sigma in SIGMA_GRID:
            for n_train in NTRAIN_GRID:
                for seed in SEEDS:
                    rng = np.random.default_rng(1000 * seed + 7)
                    # H-invariant class templates (regenerated per seed)
                    T = rng.standard_normal((2, ORDER)) @ Pi_true.T
                    T /= np.linalg.norm(T, axis=1, keepdims=True)
                    TEMPLATES = T
                    label_signal_scale = float(np.linalg.norm(T[1] - T[0]))

                    # test set depends only on (seed, sigma) so it is IDENTICAL across n_train
                    trng = np.random.default_rng(500000 + 1000 * seed + int(10 * sigma))
                    Xte, yte, _ = make_data(H, sigma, N_TEST, trng)
                    drng = np.random.default_rng(700000 + 1000 * seed + int(10 * sigma))
                    Xtr, ytr, _ = make_data(H, sigma, n_train, drng)

                    frng = np.random.default_rng(99000 + seed)
                    acc_or, l2_or = run_fixed_arm(Xtr, ytr, Xte, yte, uniform_p_on(H), frng)
                    acc_wr, l2_wr = run_fixed_arm(Xtr, ytr, Xte, yte, uniform_p_on(Hw), frng)
                    p_triv = np.zeros(ORDER); p_triv[0] = 1.0
                    acc_tr, l2_tr = run_fixed_arm(Xtr, ytr, Xte, yte, p_triv, frng)
                    acc_un, l2_un = run_fixed_arm(Xtr, ytr, Xte, yte, np.ones(ORDER) / ORDER, frng)
                    acc_di, p_learned, l2_di = run_discovered_arm(Xtr, ytr, Xte, yte, frng)

                    mass_learned = float(p_learned[hidx].sum())
                    mrand = random_frame_mass(H, np.random.default_rng(31337 + seed))
                    mass_random = float(mrand.mean())

                    rows.append(dict(
                        true_subgroup=true_name, true_order=len(H),
                        wrong_subgroup=wrong_name, wrong_order=len(Hw),
                        sigma=sigma, n_train=n_train, seed=seed,
                        is_primary_cell=int((sigma, n_train) == PRIMARY_CELL),
                        label_signal_scale=label_signal_scale,
                        mass_H_learned=mass_learned,
                        mass_H_random=mass_random,
                        mass_H_analytic=len(H) / ORDER,
                        recovery_gap=mass_learned - mass_random,
                        acc_discovered=float(acc_di), acc_oracle=float(acc_or),
                        acc_wrong=float(acc_wr), acc_trivial=float(acc_tr),
                        acc_uniform_ambient=float(acc_un),
                        l2_discovered=l2_di, l2_oracle=l2_or, l2_wrong=l2_wr,
                        l2_trivial=l2_tr, l2_uniform_ambient=l2_un,
                    ))
        print(f"[u2] done true_subgroup={true_name}", flush=True)

    path = os.path.join(OUT, "u2_symmetry_discovery.csv")
    with open(path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"[u2] wrote {path}  ({len(rows)} rows)")
    summarize(rows)


def summarize(rows):
    def se(a):
        a = np.asarray(a, float)
        return float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float("nan")

    keys = sorted({(r["true_subgroup"], r["sigma"], r["n_train"]) for r in rows})
    out = []
    for tname, sigma, n_train in keys:
        sub = [r for r in rows if (r["true_subgroup"], r["sigma"], r["n_train"]) == (tname, sigma, n_train)]
        gap = [r["recovery_gap"] for r in sub]
        d_or = [r["acc_oracle"] - r["acc_discovered"] for r in sub]
        d_wr = [r["acc_discovered"] - r["acc_wrong"] for r in sub]
        d_tr = [r["acc_discovered"] - r["acc_trivial"] for r in sub]
        d_disc = [r["acc_oracle"] - r["acc_trivial"] for r in sub]
        lss = min(r["label_signal_scale"] for r in sub)

        # guards FIRST, exactly as pre-registered
        if lss < VACUITY_TOL:
            verdict = "vacuous_excluded"
        elif not (np.mean(d_disc) > 2 * se(d_disc)):
            verdict = "nondiscriminating_excluded"
        else:
            a = np.mean(gap) > 2 * se(gap)
            b = np.mean(d_or) < TOL_ORACLE
            c = np.mean(d_wr) > 2 * se(d_wr)
            no_gain = not (np.mean(d_tr) > 2 * se(d_tr))
            if a and b and c:
                verdict = "CONFIRMS"
            elif (not a) or no_gain:
                verdict = "KILLS"
            else:
                verdict = "PARTIAL"
        out.append(dict(
            true_subgroup=tname, sigma=sigma, n_train=n_train,
            is_primary_cell=int((sigma, n_train) == PRIMARY_CELL),
            n_seeds=len(sub), min_label_signal_scale=lss,
            mean_mass_H_learned=float(np.mean([r["mass_H_learned"] for r in sub])),
            mean_mass_H_random=float(np.mean([r["mass_H_random"] for r in sub])),
            mean_recovery_gap=float(np.mean(gap)), se_recovery_gap=se(gap),
            mean_acc_discovered=float(np.mean([r["acc_discovered"] for r in sub])),
            se_acc_discovered=se([r["acc_discovered"] for r in sub]),
            mean_acc_oracle=float(np.mean([r["acc_oracle"] for r in sub])),
            se_acc_oracle=se([r["acc_oracle"] for r in sub]),
            mean_acc_wrong=float(np.mean([r["acc_wrong"] for r in sub])),
            mean_acc_trivial=float(np.mean([r["acc_trivial"] for r in sub])),
            mean_acc_uniform_ambient=float(np.mean([r["acc_uniform_ambient"] for r in sub])),
            mean_oracle_minus_discovered=float(np.mean(d_or)),
            mean_discovered_minus_wrong=float(np.mean(d_wr)), se_discovered_minus_wrong=se(d_wr),
            mean_discovered_minus_trivial=float(np.mean(d_tr)), se_discovered_minus_trivial=se(d_tr),
            mean_oracle_minus_trivial=float(np.mean(d_disc)), se_oracle_minus_trivial=se(d_disc),
            verdict=verdict,
        ))
    path = os.path.join(OUT, "u2_symmetry_discovery_summary.csv")
    with open(path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wtr.writeheader()
        wtr.writerows(out)
    print(f"[u2] wrote {path}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
