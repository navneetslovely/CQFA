
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from datetime import datetime

import numpy as np

from cqfa import group as G
from cqfa.dynamical_lie_algebra import (g_purity, hea_circuit_generators, restrict)
from cqfa.simulator import cnot_ring, embed_1q
from cqfa.group import X as PX, Y as PY, Z as PZ
from experiments.trainability_law import trainability_law as T

TARGET_TRAIN_LOSS = 0.30          # pre-registered, do not change
VACUITY_TOL = 1e-10               # pre-registered

GRAD_ZERO_TOL = 1e-18


# ===========================================================================
# circuit engine: ops = list of ("param", H_hermitian, (w, V)) or ("fixed", U)
# exact adjoint gradients, no finite differences
# ===========================================================================
def _herm_eig(H):
    return np.linalg.eigh((H + H.conj().T) / 2.0)


def _apply_param(op, theta, psi):
    """exp(-i theta H) @ psi via cached eigendecomposition (2 matvecs)."""
    w, V = op[2]
    return V @ (np.exp(-1j * theta * w) * (V.conj().T @ psi))


def build_equivariant_ops(gens, depth):
    eigs = [_herm_eig(H) for H in gens]
    ops = []
    for _ in range(depth):
        for H, e in zip(gens, eigs):
            ops.append(("param", H, e))
    return ops


def build_generic_hea_ops(n, depth):
    """D layers of [RY_q, RZ_q for all q] then a FIXED CNOT ring. DLA = su(2^n)."""
    E = cnot_ring(n) if n > 1 else np.eye(2, dtype=complex)
    ops = []
    for _ in range(depth):
        for q in range(n):
            for Pmat in (PY, PZ):
                H = embed_1q(Pmat, q, n)
                ops.append(("param", H, _herm_eig(H)))
        if n > 1:
            ops.append(("fixed", E, None))
    return ops


def n_params(ops):
    return sum(1 for o in ops if o[0] == "param")


def val_grad_multi(ops, thetas, psi, O_list):

    states = [psi.astype(complex)]
    t = 0
    for op in ops:
        if op[0] == "param":
            states.append(_apply_param(op, thetas[t], states[-1]))
            t += 1
        else:
            states.append(op[1] @ states[-1])
    y = states[-1]
    # Phi: (dim, n_obs) -- all observables carried through the backward pass together
    Phi = np.column_stack([O @ y for O in O_list])
    vals = np.real(np.einsum("i,ik->k", y.conj(), Phi))

    P = t
    grads = np.zeros((P, len(O_list)))
    k = P - 1
    for idx in range(len(ops) - 1, -1, -1):
        op = ops[idx]
        psi_out = states[idx + 1]
        if op[0] == "param":
            # d/dtheta <psi_out| ... = 2 Re[ Phi^dag (-i H) psi_out ]
            grads[k] = 2.0 * np.real(np.einsum("ik,i->k", Phi.conj(),
                                               -1j * (op[1] @ psi_out)))
            w, V = op[2]
            Phi = V @ (np.exp(1j * thetas[k] * w)[:, None] * (V.conj().T @ Phi))
            k -= 1
        else:
            Phi = op[1].conj().T @ Phi
    return vals, grads


def forward_batch(ops, thetas, Psi):
    """Batched forward: Psi is (dim, m). Returns list of states for the adjoint."""
    states = [Psi.astype(complex)]
    t = 0
    for op in ops:
        if op[0] == "param":
            w, V = op[2]
            states.append(V @ (np.exp(-1j * thetas[t] * w)[:, None] * (V.conj().T @ states[-1])))
            t += 1
        else:
            states.append(op[1] @ states[-1])
    return states


def val_grad_batch(ops, thetas, Psi, O):
    """Exact per-sample values (m,) and gradients (P, m) for a batch of inputs."""
    states = forward_batch(ops, thetas, Psi)
    Y = states[-1]
    Phi = O @ Y
    vals = np.real(np.einsum("im,im->m", Y.conj(), Phi))
    P = sum(1 for o in ops if o[0] == "param")
    grads = np.zeros((P, Psi.shape[1]))
    k = P - 1
    for idx in range(len(ops) - 1, -1, -1):
        op = ops[idx]
        psi_out = states[idx + 1]
        if op[0] == "param":
            grads[k] = 2.0 * np.real(np.einsum("im,im->m", Phi.conj(),
                                               -1j * (op[1] @ psi_out)))
            w, V = op[2]
            Phi = V @ (np.exp(1j * thetas[k] * w)[:, None] * (V.conj().T @ Phi))
            k -= 1
        else:
            Phi = op[1].conj().T @ Phi
    return vals, grads


# ===========================================================================
# observables, inputs, normalisation
# ===========================================================================
def _traceless(H):
    d = H.shape[0]
    return H - (np.trace(H) / d) * np.eye(d, dtype=complex)


def _hs(M):
    return float(np.sqrt(np.real(np.vdot(M.reshape(-1), M.reshape(-1)))))


def prepare_observable(M_raw):
    """Traceless-project, record the raw HS norm, then HS-normalise.
    Returns (O, raw_hs_norm, op_norm) with O = None if the raw operator vanishes."""
    Mt = _traceless(np.asarray(M_raw, dtype=complex))
    raw = _hs(Mt)
    if raw < VACUITY_TOL:
        return None, raw, 0.0
    O = Mt / raw
    O = (O + O.conj().T) / 2.0
    return O, raw, float(np.linalg.norm(O, 2))


def _embed_pauli_string(paulis, n):
    """Kron a dict {qubit: 2x2} into an n-qubit operator."""
    out = np.array([[1.0 + 0j]])
    for i in range(n):
        out = np.kron(out, paulis.get(i, np.eye(2, dtype=complex)))
    return out


def equivariant_observables(pack, n):
  
    dH = pack["dim_hplus"]
    Bplus = pack["Bplus"]
    out = {}

    A = pack["basis"][0].reshape(dH, dH)
    dla_h = (-1j * A + (-1j * A).conj().T) / 2.0
    out["dla_elem"] = (dla_h, "global_in_algebra")

    e0 = np.zeros((dH, dH), dtype=complex)
    e0[0, 0] = 1.0
    out["proj_rank1"] = (e0, "global")

    out["X0_local"] = (restrict(_embed_pauli_string({0: PX}, n), Bplus), "1_local")
    if n >= 3:
        out["Z1Z2_local"] = (restrict(_embed_pauli_string({1: PZ, 2: PZ}, n), Bplus), "2_local")
    out["Z0_local"] = (restrict(_embed_pauli_string({0: PZ}, n), Bplus), "1_local")
    if n >= 2:
        out["Z0Z1_local"] = (restrict(_embed_pauli_string({0: PZ, 1: PZ}, n), Bplus), "2_local")
    return out


def haar_state(dim, rng):
    v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return v / np.linalg.norm(v)


def haar_batch(dim, m, rng):
    Xm = rng.standard_normal((dim, m)) + 1j * rng.standard_normal((dim, m))
    return Xm / np.linalg.norm(Xm, axis=0, keepdims=True)


def gpurity_maximal_state(pack, rng, iters=400, lr=0.5):
   
    dH = pack["dim_hplus"]
    basis = pack["basis"]
    Bs = [basis[j].reshape(dH, dH) for j in range(basis.shape[0])]
    chi = haar_state(dH, rng)
    best, best_chi = -1.0, chi
    for _ in range(iters):
        cs = np.array([chi.conj() @ (B @ chi) for B in Bs])
        f = float(np.sum(np.abs(cs) ** 2))
        if f > best:
            best, best_chi = f, chi.copy()
        grad = np.zeros(dH, dtype=complex)
        for c, B in zip(cs, Bs):
            grad += np.conj(c) * (B @ chi) + c * (B.conj().T @ chi)
        grad = grad - (chi.conj() @ grad) * chi        # project onto tangent space
        nrm = np.linalg.norm(grad)
        if nrm < 1e-14:
            break
        chi = chi + lr * grad / nrm
        chi = chi / np.linalg.norm(chi)
    return best_chi


def draw_input(kind, pack, rng):
    if kind == "haar_Hp":
        return haar_state(pack["dim_hplus"], rng)
    if kind == "aligned_gG":
        return T.aligned_input(pack, rng)
    if kind == "gpure_max":
        return gpurity_maximal_state(pack, rng)
    raise ValueError(kind)


# ===========================================================================
# BLOCK 1: gradient variance across depth, parameter position, observable, input
# ===========================================================================
def param_positions(P, L):
    """Four parameter locations: first / quarter / mid / last, with their layer."""
    idx = {"first": 0, "quarter": max(0, P // 4), "mid": P // 2, "last": P - 1}
    out = {}
    for name, k in idx.items():
        out[name] = (int(k), int(k // L) + 1, float((k + 1) / P))
    return out


def grad_variance_cells(arm, n, depth, pack, gens, obs_specs, input_kinds,
                        seeds, S, rng_base, extra=None):
    """One (arm, n, depth) configuration -> rows for every
    (input_dist, observable, param_position) combination."""
    ops = build_equivariant_ops(gens, depth) if arm == "cqfa_equivariant" \
        else build_generic_hea_ops(n, depth)
    P = n_params(ops)
    L = len(gens) if arm == "cqfa_equivariant" else 2 * n
    positions = param_positions(P, L)

    names, Olist, meta = [], [], []
    for oname, (raw, locality) in obs_specs.items():
        O, raw_hs, op_nrm = prepare_observable(raw)
        pg = 0.0 if O is None else (g_purity(O, pack["basis"]) if pack is not None else float("nan"))
        names.append(oname)
        meta.append(dict(locality=locality, raw_hs=raw_hs, op_norm=op_nrm, p_g_O=pg,
                         vacuous=1 if (O is None or (pack is not None and pg < VACUITY_TOL)) else 0))
        Olist.append(np.zeros((raw.shape[0], raw.shape[0]), dtype=complex) if O is None else O)

    rows = []
    for ikind in input_kinds:
        # per-seed variance estimates, then mean +/- std across seeds
        per_seed = {(oname, pos): [] for oname in names for pos in positions}
        pg_rho_seeds = []
        for sd in seeds:
            rng = np.random.default_rng(rng_base + 7919 * sd)
            if arm == "cqfa_equivariant":
                chi = draw_input(ikind, pack, rng)
                pg_rho_seeds.append(g_purity(np.outer(chi, chi.conj()), pack["basis"]))
            else:
                chi = haar_state(2 ** n, rng)
                pg_rho_seeds.append(float("nan"))
            gsamp = np.empty((S, P, len(names)))
            for s in range(S):
                th = rng.uniform(0, 2 * np.pi, P)
                _, gr = val_grad_multi(ops, th, chi, Olist)
                gsamp[s] = gr
            for oi, oname in enumerate(names):
                for pos, (k, _, _) in positions.items():
                    per_seed[(oname, pos)].append(float(np.var(gsamp[:, k, oi])))

        for oi, oname in enumerate(names):
            m = meta[oi]
            for pos, (k, layer, frac) in positions.items():
                vals = np.array(per_seed[(oname, pos)])
                mean_v = float(np.mean(vals))
                lo, hi = _boot_ci(vals)
                # vacuity guard, applied per CELL and BEFORE any test
                if m["raw_hs"] < VACUITY_TOL:
                    reason = "observable_identically_zero_on_Hplus"
                elif m["p_g_O"] < VACUITY_TOL and pack is not None:
                    reason = "observable_orthogonal_to_gG"
                elif mean_v < GRAD_ZERO_TOL:
                    reason = "gradient_identically_zero_at_this_parameter_position"
                else:
                    reason = ""
                rows.append(dict(
                    block="grad_variance", arm=arm,
                    group_family=(extra or {}).get("group_family", ""),
                    n_qubits=n, dim_full=2 ** n,
                    dim_hplus=(pack["dim_hplus"] if pack is not None else 2 ** n),
                    dim_gG=(pack["d_eff"] if pack is not None else 4 ** n - 1),
                    depth=depth, n_generators_per_layer=L, n_params=P,
                    gate_order="fixed_cyclic_layer_repeat",
                    param_draw="iid_uniform_0_2pi",
                    input_distribution=ikind,
                    observable=oname, cost_locality=m["locality"],
                    obs_hs_norm=(0.0 if m["raw_hs"] < VACUITY_TOL else 1.0),
                    obs_raw_hs_norm=m["raw_hs"], obs_op_norm=m["op_norm"],
                    p_g_O=m["p_g_O"], p_g_rho=float(np.mean(pg_rho_seeds)),
                    param_position=pos, param_index=k, param_layer=layer,
                    param_depth_frac=frac,
                    n_seeds=len(seeds), samples_per_seed=S,
                    grad_var_mean=mean_v, grad_var_std=float(np.std(vals)),
                    grad_var_ci_lo=lo, grad_var_ci_hi=hi,
                    grad_var_times_dim_hplus=(mean_v * pack["dim_hplus"]) if pack is not None else "",
                    grad_var_times_4n=mean_v * (4 ** n),
                    label_signal_scale=m["p_g_O"] * m["raw_hs"],
                    vacuous_excluded=(1 if reason else 0),
                    vacuity_reason=reason,
                ))
    return rows


def _boot_ci(vals, B=4000, alpha=0.05, seed=12345):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    bs = rng.choice(vals, size=(B, len(vals)), replace=True).mean(axis=1)
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


# ===========================================================================
# BLOCK 2: OPTIMISATION SUCCESS  (the key new measurement)
# ===========================================================================
def make_task(pack, rng, n_train=100, n_test=200):
    """Binary classification with a REALISABLE teacher: T is a g_G element, so
    the equivariant ansatz can in principle represent the labelling."""
    dH = pack["dim_hplus"]
    j = 1 if pack["basis"].shape[0] > 1 else 0
    Tm = -1j * pack["basis"][j].reshape(dH, dH)
    Tm = (Tm + Tm.conj().T) / 2.0
    Xtr = haar_batch(dH, n_train, rng)
    Xte = haar_batch(dH, n_test, rng)
    ftr = np.real(np.einsum("im,ij,jm->m", Xtr.conj(), Tm, Xtr))
    fte = np.real(np.einsum("im,ij,jm->m", Xte.conj(), Tm, Xte))
    return Xtr, (ftr > 0).astype(float), Xte, (fte > 0).astype(float), float(np.std(ftr))


def train_one(ops, O, Xtr, ytr, Xte, yte, theta0, epochs, lr=0.05):
    """Full-batch Adam on the mean logistic loss of sigmoid(a*f(theta,x)+b).
    Returns the loss trajectory, final losses/accuracies and the initial gradient."""
    theta = theta0.copy()
    a, b = 2.0, 0.0
    P = len(theta)
    m1 = np.zeros(P + 2); m2 = np.zeros(P + 2)
    b1, b2, eps = 0.9, 0.999, 1e-8
    traj = np.empty(epochs)
    g0 = None
    for ep in range(epochs):
        F, GR = val_grad_batch(ops, theta, Xtr, O)      # F: (m,), GR: (P, m)
        z = a * F + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        loss = float(np.mean(-(ytr * np.log(p + 1e-12) + (1 - ytr) * np.log(1 - p + 1e-12))))
        traj[ep] = loss
        dz = (p - ytr) / len(ytr)
        gth = GR @ (dz * a)
        ga = float(np.dot(dz, F)); gb = float(np.sum(dz))
        if ep == 0:
            g0 = gth.copy()
        gall = np.concatenate([gth, [ga, gb]])
        m1 = b1 * m1 + (1 - b1) * gall
        m2 = b2 * m2 + (1 - b2) * gall ** 2
        mh = m1 / (1 - b1 ** (ep + 1)); vh = m2 / (1 - b2 ** (ep + 1))
        step = lr * mh / (np.sqrt(vh) + eps)
        theta -= step[:P]; a -= float(step[P]); b -= float(step[P + 1])

    def _acc(X, y):
        F, _ = val_grad_batch(ops, theta, X, O)
        return float(np.mean(((a * F + b) > 0).astype(float) == y))
    Ftr, _ = val_grad_batch(ops, theta, Xtr, O)
    z = a * Ftr + b
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    final_loss = float(np.mean(-(ytr * np.log(p + 1e-12) + (1 - ytr) * np.log(1 - p + 1e-12))))
    return dict(traj=traj, final_train_loss=final_loss,
                train_acc=_acc(Xtr, ytr), test_acc=_acc(Xte, yte),
                init_grad_sq_mean=float(np.mean(g0 ** 2)),
                init_grad_var=float(np.var(g0)))


def optimisation_cells(n, depth, pack, gens, restarts, init_scales, epochs, rng_base):
    ops = build_equivariant_ops(gens, depth)
    P = n_params(ops)
    L = len(gens)
    dH = pack["dim_hplus"]
   
    O_hs, raw_hs, op_nrm = prepare_observable(equivariant_observables(pack, n)["dla_elem"][0])
    pgO = g_purity(O_hs, pack["basis"])
    O = O_hs / op_nrm                       # unit operator norm

    rows = []
    for scale in init_scales:
        finals, tests, iters, succ, ig = [], [], [], [], []
        capped = 0
        sig = None
        for r in range(restarts):
            rng = np.random.default_rng(rng_base + 104729 * r + int(scale * 1000))
            Xtr, ytr, Xte, yte, sigma = make_task(pack, rng)
            sig = sigma
            theta0 = rng.uniform(0, 2 * np.pi, P) * scale
            res = train_one(ops, O, Xtr, ytr, Xte, yte, theta0, epochs)
            finals.append(res["final_train_loss"])
            tests.append(res["test_acc"])
            ig.append(res["init_grad_sq_mean"])
            hit = np.nonzero(res["traj"] <= TARGET_TRAIN_LOSS)[0]
            if len(hit):
                iters.append(int(hit[0]) + 1)
            else:
                capped += 1
            succ.append(1.0 if res["final_train_loss"] <= TARGET_TRAIN_LOSS else 0.0)

        vac = 1 if (sig is None or sig < VACUITY_TOL or pgO < VACUITY_TOL) else 0
        rows.append(dict(
            block="optimisation", arm="cqfa_equivariant", group_family="Z2-global",
            n_qubits=n, dim_full=2 ** n, dim_hplus=dH, dim_gG=pack["d_eff"],
            depth=depth, n_generators_per_layer=L, n_params=P,
            gate_order="fixed_cyclic_layer_repeat", param_draw="iid_uniform_0_2pi",
            input_distribution="haar_Hp", observable="dla_elem",
            cost_locality="global_in_algebra", obs_raw_hs_norm=raw_hs, obs_op_norm=op_nrm,
            p_g_O=pgO, train_obs_normalisation="unit_operator_norm",
            gradvar_obs_normalisation="unit_hs_norm",
            optimiser="adam", lr=0.05, epochs=epochs, init_scale=scale,
            n_restarts=restarts, target_train_loss=TARGET_TRAIN_LOSS,
            final_train_loss_mean=float(np.mean(finals)),
            final_train_loss_std=float(np.std(finals)),
            final_test_acc_mean=float(np.mean(tests)),
            final_test_acc_std=float(np.std(tests)),
            success_fraction=float(np.mean(succ)),
            n_restarts_capped=capped,
            iters_to_target_mean=(float(np.mean(iters)) if iters else ""),
            iters_to_target_lower_bound=(epochs if capped else ""),
            init_grad_sq_mean=float(np.mean(ig)),
            init_grad_sq_std=float(np.std(ig)),
            label_signal_scale=(sig if sig is not None else 0.0),
            vacuous_excluded=vac,
        ))
    return rows


# ===========================================================================
# fitting: log-log exponent with uncertainty
# ===========================================================================
def fit_loglog(x, y):
    """OLS of log10(y) on log10(x). Returns slope, stderr, 95% CI, R2."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return None
    lx, ly = np.log10(x), np.log10(y)
    A = np.column_stack([lx, np.ones_like(lx)])
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    resid = ly - A @ coef
    dof = len(x) - 2
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    se = float(np.sqrt(cov[0, 0]))
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}.get(dof, 2.0)
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return dict(slope=float(coef[0]), intercept=float(coef[1]), se=se,
                ci_lo=float(coef[0] - tcrit * se), ci_hi=float(coef[0] + tcrit * se),
                r2=r2, n_points=int(len(x)))


def round_to_unc(val, unc):
    """Report a fitted value to no more digits than the uncertainty supports."""
    if not np.isfinite(unc) or unc <= 0:
        return f"{val:.3f}"
    d = max(0, int(np.ceil(-np.log10(unc))) + 1)
    return f"{val:.{d}f} +/- {unc:.{d}f}"


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / den) if den > 0 else float("nan")


# ===========================================================================
# drivers
# ===========================================================================
def _write(path, rows):
    if not rows:
        print("  (no rows for", path, ")")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"wrote {path} ({len(rows)} rows, {len(keys)} cols)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8])
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--opt-sizes", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    ap.add_argument("--opt-depths", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--samples", type=int, default=250)
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out", default="results/trainability_law")
    ap.add_argument("--tag", default="trainability_full")
    args = ap.parse_args(argv)

    t0 = time.time()
    rows = []
    input_kinds = ["haar_Hp", "aligned_gG", "gpure_max"]

    # ---- BLOCK 1a: CQFA equivariant arm -----------------------------------
    print("=== BLOCK 1a: equivariant gradient variance (depth x obs x input x position) ===")
    packs = {}
    for n in args.sizes:
        rep = G.z2_pauli(n)
        pack = T.dla_pack(rep, n)
        if pack is None:
            continue
        gens = T.equivariant_ansatz_generators(rep, n, pack)
        if not gens:
            continue
        packs[n] = (pack, gens)
        obs = equivariant_observables(pack, n)
        for depth in args.depths:
            r = grad_variance_cells("cqfa_equivariant", n, depth, pack, gens, obs,
                                    input_kinds, args.seeds, args.samples,
                                    rng_base=20000 + 100 * n + depth,
                                    extra=dict(group_family="Z2-global"))
            rows += r
            sel = [x for x in r if x["observable"] == "dla_elem"
                   and x["input_distribution"] == "haar_Hp" and x["param_position"] == "mid"]
            if sel:
                s = sel[0]
                print(f"  n={n} dimH+={s['dim_hplus']:>4} D={depth} P={s['n_params']:>3} "
                      f"Var(mid,dla,haar)={s['grad_var_mean']:.3e} "
                      f"Var*dimH+={s['grad_var_times_dim_hplus']:.4f} "
                      f"P_g(rho)={s['p_g_rho']:.3e}")

    # ---- BLOCK 1b: generic HEA reference arm (the 1/4^n baseline) ---------
    print("\n=== BLOCK 1b: generic HEA reference (strong same-depth baseline) ===")
    for n in args.sizes:
        proj0 = np.zeros((2 ** n, 2 ** n), dtype=complex)
        proj0[0, 0] = 1.0                      # |0><0|^{(x)n}: the global cost
        obs_g = {
            "proj_rank1_full": (proj0, "global"),
            "Z_local_full": (_embed_pauli_string({0: PZ}, n), "1_local"),
        }
        for depth in args.depths:
            r = grad_variance_cells("generic_hea", n, depth, None, [None] * (2 * n), obs_g,
                                    ["haar_full"], args.seeds, args.samples,
                                    rng_base=40000 + 100 * n + depth,
                                    extra=dict(group_family="none"))
            rows += r
            sel = [x for x in r if x["observable"] == "proj_rank1_full"
                   and x["param_position"] == "mid"]
            if sel:
                s = sel[0]
                print(f"  n={n} dim={2**n:>4} D={depth} P={s['n_params']:>3} "
                      f"Var(mid,global)={s['grad_var_mean']:.3e} "
                      f"Var*4^n={s['grad_var_times_4n']:.4f} "
                      f"[Ragone 1/4^n = {T.ragone_var_generic(n):.3e}]")

    # ---- BLOCK 2: optimisation success ------------------------------------
    print("\n=== BLOCK 2: OPTIMISATION SUCCESS (train to convergence) ===")
    for n in args.opt_sizes:
        if n not in packs:
            continue
        pack, gens = packs[n]
        for depth in args.opt_depths:
            r = optimisation_cells(n, depth, pack, gens, args.restarts,
                                   [1.0, 0.1], args.epochs,
                                   rng_base=60000 + 100 * n + depth)
            rows += r
            for s in r:
                print(f"  n={n} dimH+={s['dim_hplus']:>4} D={depth} init={s['init_scale']:.1f} "
                      f"loss={s['final_train_loss_mean']:.3f} test={s['final_test_acc_mean']:.3f} "
                      f"succ={s['success_fraction']:.2f} "
                      f"iters={s['iters_to_target_mean'] if s['iters_to_target_mean'] != '' else 'CAP'} "
                      f"|g0|^2={s['init_grad_sq_mean']:.2e}")

    out_main = os.path.join(args.out, f"{args.tag}.csv")
    _write(out_main, rows)

    # ---- fits --------------------------------------------------------------
    print("\n=== FITS: exponent of Var ~ dim^b, per subset ===")
    fits = []
    gv = [r for r in rows if r["block"] == "grad_variance" and r["vacuous_excluded"] == 0]

    for arm, xkey in (("cqfa_equivariant", "dim_hplus"),
                      ("cqfa_equivariant", "dim_gG"),
                      ("generic_hea", "dim_full")):
        subs = sorted({(r["depth"], r["observable"], r["input_distribution"], r["param_position"])
                       for r in gv if r["arm"] == arm})
        for (d, ob, ik, pos) in subs:
            cell = [r for r in gv if r["arm"] == arm and r["depth"] == d
                    and r["observable"] == ob and r["input_distribution"] == ik
                    and r["param_position"] == pos]
            if len(cell) < 3:
                continue
            f = fit_loglog([c[xkey] for c in cell], [c["grad_var_mean"] for c in cell])
            if f is None:
                continue
            prod = np.array([c["grad_var_mean"] * c[xkey] for c in cell], float)
            fits.append(dict(arm=arm, x_variable=xkey, depth=d, observable=ob,
                             cost_locality=cell[0]["cost_locality"],
                             input_distribution=ik, param_position=pos,
                             n_points=f["n_points"], slope=f["slope"], slope_se=f["se"],
                             slope_ci_lo=f["ci_lo"], slope_ci_hi=f["ci_hi"], r2=f["r2"],
                             slope_reported=round_to_unc(f["slope"], f["ci_hi"] - f["slope"]),
                             ci_contains_minus1=int(f["ci_lo"] <= -1.0 <= f["ci_hi"]),
                             const_var_times_dim_mean=float(np.mean(prod)),
                             const_var_times_dim_std=float(np.std(prod))))
            print(f"  [{arm:17s}] D={d} {ob:16s} {ik:11s} {pos:8s} "
                  f"b = {fits[-1]['slope_reported']}  R2={f['r2']:.3f} "
                  f"{'CI covers -1' if fits[-1]['ci_contains_minus1'] else '*** CI EXCLUDES -1 ***'}")
    _write(os.path.join(args.out, f"{args.tag}_fits.csv"), fits)

    # ---- regimes + verdict -------------------------------------------------
    print("\n=== FOUR-REGIME CLASSIFICATION + PRE-REGISTERED VERDICT ===")
    reg = []

    def _pick(arm, ob, ik, xk):
        return [f for f in fits if f["arm"] == arm and f["observable"] == ob
                and f["input_distribution"] == ik and f["x_variable"] == xk]

    for label, arm, ob, ik, xk, note in [
        ("R1_global_barren_plateau", "generic_hea", "proj_rank1_full", "haar_full", "dim_full",
         "generic HEA + global rank-1 cost + Haar inputs (the 1/4^n reference)"),
        ("R2_local_cost_scaling", "generic_hea", "Z_local_full", "haar_full", "dim_full",
         "generic HEA + 1-local cost + Haar inputs (Cerezo local-cost regime)"),
        ("R3_symmetry_restricted_concentration", "cqfa_equivariant", "dla_elem", "haar_Hp",
         "dim_hplus", "equivariant CQFA + in-algebra global observable + Haar-on-H_+ inputs"),
        ("R3b_symmetry_restricted_vs_dim_g", "cqfa_equivariant", "dla_elem", "haar_Hp",
         "dim_gG", "same cells, fitted against dim(g_G) instead of dim H_+"),
        ("R2b_symmetry_restricted_1local", "cqfa_equivariant", "X0_local", "haar_Hp",
         "dim_hplus", "equivariant CQFA + reachable 1-local cost (X_0)"),
        ("R2c_symmetry_restricted_2local", "cqfa_equivariant", "Z1Z2_local", "haar_Hp",
         "dim_hplus", "equivariant CQFA + reachable 2-local cost (Z_1 Z_2)"),
        ("R4_input_dependence_gpuremax", "cqfa_equivariant", "dla_elem", "gpure_max",
         "dim_hplus", "equivariant CQFA + g-purity-MAXIMISING inputs (is the law input-driven?)"),
        ("R4b_input_dependence_aligned", "cqfa_equivariant", "dla_elem", "aligned_gG",
         "dim_hplus", "equivariant CQFA + aligned inputs (the distribution used in prior CSVs)"),
    ]:
        sel = _pick(arm, ob, ik, xk)
        for f in sel:
            reg.append(dict(regime=label, description=note, arm=arm, observable=ob,
                            cost_locality=f["cost_locality"], input_distribution=ik,
                            depth=f["depth"], param_position=f["param_position"],
                            x_variable=xk, slope=f["slope"],
                            slope_ci_lo=f["slope_ci_lo"], slope_ci_hi=f["slope_ci_hi"],
                            slope_reported=f["slope_reported"], r2=f["r2"],
                            ci_contains_minus1=f["ci_contains_minus1"]))

    # (K1) does the exponent depend on unspecified choices?
    core = [f for f in fits if f["arm"] == "cqfa_equivariant"
            and f["observable"] == "dla_elem" and f["x_variable"] == "dim_hplus"]
    slopes = np.array([f["slope"] for f in core], float)
    halfw = np.array([f["slope_ci_hi"] - f["slope"] for f in core], float)
    spread = float(slopes.max() - slopes.min()) if len(slopes) else float("nan")
    med_hw = float(np.median(halfw)) if len(halfw) else float("nan")
    n_excl = int(sum(1 for f in core if not f["ci_contains_minus1"]))
    k1_triggered = int((spread > med_hw) or (n_excl > 0))
    print(f"  (K1) exponent spread across subsets = {spread:.3f}; "
          f"median CI half-width = {med_hw:.3f}; subsets whose CI excludes -1 = {n_excl}/{len(core)} "
          f"-> K1 {'TRIGGERED' if k1_triggered else 'not triggered'}")

    # (S2/K2) does optimisation success track initial gradient variance?
    opt = [r for r in rows if r["block"] == "optimisation" and r["vacuous_excluded"] == 0]
    ver = []
    for scale in (1.0, 0.1):
        sub = [r for r in opt if r["init_scale"] == scale]
        if len(sub) < 3:
            continue
        rho_succ = spearman([r["init_grad_sq_mean"] for r in sub],
                            [r["success_fraction"] for r in sub])
        rho_acc = spearman([r["init_grad_sq_mean"] for r in sub],
                           [r["final_test_acc_mean"] for r in sub])
        rho_dim = spearman([r["dim_hplus"] for r in sub],
                           [r["success_fraction"] for r in sub])
        k2 = int(not (rho_succ > 0))
        ver.append(dict(regime="verdict_optimisation_tracks_variance",
                        description=f"init_scale={scale}: Spearman(init |g|^2, success_fraction)",
                        arm="cqfa_equivariant", observable="dla_elem",
                        input_distribution="haar_Hp", depth="all", param_position="all",
                        x_variable="init_grad_sq_mean",
                        spearman_initgrad_vs_success=rho_succ,
                        spearman_initgrad_vs_testacc=rho_acc,
                        spearman_dimhplus_vs_success=rho_dim,
                        n_cells=len(sub), init_scale=scale, k2_triggered=k2))
        print(f"  (S2/K2) init_scale={scale}: rho(|g0|^2, success)={rho_succ:+.3f}  "
              f"rho(|g0|^2, test_acc)={rho_acc:+.3f}  rho(dimH+, success)={rho_dim:+.3f} "
              f"-> K2 {'TRIGGERED' if k2 else 'not triggered'}")
    reg += ver

    k2_any = int(any(v["k2_triggered"] for v in ver)) if ver else 1
    verdict = ("SURVIVES" if (not k1_triggered and not k2_any)
               else "DEMOTED_ensemble_specific" if (k1_triggered and not k2_any)
               else "KILLED_variance_does_not_predict_trainability" if (k2_any and not k1_triggered)
               else "KILLED_both")
    reg.append(dict(regime="PREREGISTERED_VERDICT", description=verdict,
                    exponent_spread=spread, median_ci_halfwidth=med_hw,
                    n_subsets_ci_excludes_minus1=n_excl, n_subsets=len(core),
                    k1_triggered=k1_triggered, k2_triggered=k2_any))
    print(f"\n  *** PRE-REGISTERED VERDICT: {verdict} ***")

    _write(os.path.join(args.out, f"{args.tag}_regimes.csv"), reg)

    man = dict(script="trainability_full", numpy=np.__version__,
               python=platform.python_version(), args=vars(args),
               timestamp=datetime.now().isoformat(timespec="seconds"),
               target_train_loss=TARGET_TRAIN_LOSS, vacuity_tol=VACUITY_TOL,
               n_rows=len(rows), n_fits=len(fits), verdict=verdict,
               wall_s=round(time.time() - t0, 1))
    with open(os.path.join(args.out, f"{args.tag}_manifest.json"), "w") as fh:
        json.dump(man, fh, indent=2)
    print(f"\ndone in {man['wall_s']}s")


if __name__ == "__main__":
    main()
