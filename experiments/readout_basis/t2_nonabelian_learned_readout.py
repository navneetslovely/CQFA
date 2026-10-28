"""
T2 -- NON-ABELIAN learned readout vs fixed Mode F  (falsification attempt)
=========================================================================

WHY THIS TEST EXISTS
--------------------
The structural ceiling argument establishes (results/readout_dial/, results/c2_learned_frame_reduction.csv,
results/novelty_gate/) that for ABELIAN G the frame-register reduced state is circulant,
hence DFT-diagonal, hence the fixed Mode-F (character) readout is already an optimal readout:
a trainable readout can buy nothing.

That argument is WEAK for NON-ABELIAN G. For non-abelian G the Fourier transform only
BLOCK-diagonalises the frame state (one d_lambda x d_lambda block per irrep lambda), so a
fixed-basis readout returns only the DIAGONAL of each block and throws away the
d_lambda(d_lambda-1) off-diagonal real numbers. A learned readout has formal room there.
This file tests that empirically.

EXACT SETUP (statevector-exact, no shot noise, no hardware)
-----------------------------------------------------------
CQFA frame register: |Psi> = |G|^{-1/2} sum_g |g> (x) U_g|psi>, U = REGULAR representation.
The frame reduced state is the Gram matrix

        rho[g,h] = (1/|G|) <U_h psi | U_g psi> = (1/|G|) f(h^{-1} g),   f(k) = <psi|U_k|psi>

i.e. rho = (1/|G|) sum_k f(k) R_k, a G-circulant matrix (R_k = right-regular translation).
Both facts are asserted numerically at run time.

TASK (identical functional form for abelian and non-abelian G, so the comparison is clean):
  draw psi Haar-random in C^{|G|}; the label is a threshold of a LINEAR functional of the
  frame state itself,          y = 1{ Re tr(rho W) > median_train },
  for a per-seed random Hermitian W. The label is therefore, by construction, a function of
  rho alone -- the ONLY question is whether the readout preserves the information.
  Variant "generic_W": W generic Hermitian.
  Variant "offdiag_W": W is supported purely on the OFF-Fourier-diagonal part (the part a
  fixed-basis readout is blind to). For abelian G that part of rho is exactly zero, so the
  label is floating-point noise and NOBODY can learn it -- a built-in sanity control.

ARMS (matched data budget: same train/test split, same L2, same optimiser)
  * acc_modeF            fixed Mode-F readout: p = diag(F^dag rho F) in a FIXED, deterministic,
                         data-independent Fourier basis F, + trainable linear head.
                         params = |G| + 1.
  * acc_learned          trainable readout R(phi) on the frame register + linear head.
                         KEY IDENTITY (asserted numerically below): a trainable unitary basis
                         R followed by a linear head w is exactly
                             logit = sum_k w_k <R_k|rho|R_k> = tr(rho M),  M = R diag(w) R^dag,
                         and M ranges over ALL Hermitian matrices. So the learned-readout arm is
                         EXACTLY logistic regression on the full Hermitian form of rho -- a CONVEX
                         problem whose global optimum we reach. This makes the learned arm as
                         strong as it can possibly be (no local-minimum excuse). params = |G|^2 + 1.
  * acc_modeF_plus_head  CONTROL for parameter count, not information: fixed Mode-F features
                         (|G| of them) + a trainable 1-hidden-layer head with ~|G|^2 params.
                         If the learned arm's gain is merely "more parameters", this arm matches it.
  * acc_modeF_bestbasis  (supplementary) best of 8 FIXED Fourier bases, chosen by TRAIN accuracy
                         (never test) -- makes the fixed arm as strong as basis choice allows.

PRE-REGISTERED CRITERION (applied exactly as written, not tuned)
  delta_s = acc_learned(s) - acc_modeF(s) over >= 5 seeds (we use 8).
  se      = std(delta, ddof=1) / sqrt(n_seeds).
  The learned readout SURVIVES iff mean(delta) > 2 * se.  Otherwise: negative reported.
  VACUITY GUARD (applied BEFORE the delta test): if the cell's label functional is
  identically zero (label_signal_scale < 1e-10, i.e. pure floating-point roundoff),
  the cell is reported as `vacuous_excluded` regardless of delta. Without this guard
  the criterion thresholds noise against noise and can return SURVIVES on a task that
  carries no label signal at all -- which it did for C8/offdiag_W
  (delta=+0.00208, 2se=0.00137, label_signal_scale=6.98e-16) before the guard existed.

CSV: results/readout_basis/t2_nonabelian_learned_readout.csv          (per group x task x seed)
     results/readout_basis/t2_nonabelian_summary.csv                  (per group x task, verdict)
     results/readout_basis/t2_structure_check.csv                     (structural asserts / diagnostics)
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from scipy.optimize import minimize

_ROOT = os.path.abspath(__file__)
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(
        os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from cqfa.group import cyclic, dihedral4, symmetric  # noqa: E402

ROOT = os.path.abspath(__file__)
while ROOT != os.path.dirname(ROOT) and not os.path.exists(
        os.path.join(ROOT, "pyproject.toml")):
    ROOT = os.path.dirname(ROOT)
OUT = os.path.join(ROOT, "results", "readout_basis")

# ---- pre-registered hyper-parameters (fixed BEFORE looking at any accuracy) ----
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
N_TRAIN = 600
N_TEST = 600
L2 = 1e-3
MAXITER = 3000
N_RESTARTS = 5           # for the non-convex MLP control arm
N_FIXED_BASES = 8        # candidate fixed Fourier bases for acc_modeF_bestbasis
BASIS_SEED = 12345       # data-independent
CANONICAL_BASIS = 0      # acc_modeF always uses basis index 0


# ===========================================================================
# group plumbing: multiplication table, regular representation
# ===========================================================================
def mult_table(rep):
    """idx[a,b] = index of matrices[a] @ matrices[b] in matrices."""
    m = rep.order
    T = np.zeros((m, m), dtype=int)
    for a in range(m):
        for b in range(m):
            P = rep.matrices[a] @ rep.matrices[b]
            hit = [k for k in range(m) if np.max(np.abs(P - rep.matrices[k])) < 1e-9]
            assert len(hit) == 1, "group not closed / ambiguous"
            T[a, b] = hit[0]
    return T


def regular_ops(T):
    """L_a|b> = |ab> and R_a|b> = |ba> on C^{|G|}."""
    m = T.shape[0]
    L = np.zeros((m, m, m), dtype=complex)
    R = np.zeros((m, m, m), dtype=complex)
    for a in range(m):
        for b in range(m):
            L[a, T[a, b], b] = 1.0
            R[a, T[b, a], b] = 1.0
    return L, R


def is_abelian(T):
    return bool(np.array_equal(T, T.T))


# ===========================================================================
# fixed Fourier bases (deterministic, data-independent)
# ===========================================================================
def fourier_basis(L, R, b_index):
    """A canonical strong-Fourier-sampling basis of C[G].

    In the matrix-unit picture C[G] = (+)_lambda M_{d_lambda}, the left-regular algebra acts by
    LEFT multiplication and the right-regular algebra by RIGHT multiplication. A generic
    Hermitian element  H = (A + A^dag)/2,  A = sum_k c_k R_k + sum_k d_k L_k,  therefore acts as
    e -> Y e + e X with generic X, Y, whose eigenvectors are v_a w_b^T. In that eigenbasis every
    R_k (hence every G-circulant matrix, hence rho) is BLOCK-diagonal, with blocks of size
    d_lambda -- i.e. it is a valid choice of irrep bases (a Fourier basis). Deterministic.
    """
    m = L.shape[0]
    rng = np.random.default_rng(BASIS_SEED + 1000 * b_index)
    c = rng.normal(size=m) + 1j * rng.normal(size=m)
    d = rng.normal(size=m) + 1j * rng.normal(size=m)
    A = np.tensordot(c, R, axes=(0, 0)) + np.tensordot(d, L, axes=(0, 0))
    H = (A + A.conj().T) / 2.0
    w, F = np.linalg.eigh(H)
    gaps = np.diff(np.sort(w))
    assert gaps.min() > 1e-6, "degenerate spectrum -- basis not well defined"
    return F


# ===========================================================================
# frame reduced state
# ===========================================================================
def frame_state(psi, L):
    """rho[g,h] = (1/|G|) <L_h psi | L_g psi>  (exact CQFA frame reduced state)."""
    m = L.shape[0]
    V = np.einsum("gij,j->gi", L, psi)          # rows = L_g psi
    return (V @ V.conj().T) / m                 # rho[g,h] = <L_h psi | L_g psi> / |G|


# ===========================================================================
# features
# ===========================================================================
def feat_modeF(rhos, F):
    """p = diag(F^dag rho F): what a projective measurement in the fixed Fourier basis returns."""
    return np.real(np.einsum("ak,nab,bk->nk", F.conj(), rhos, F))


def feat_full(rhos):
    """All |G|^2 real d.o.f. of the Hermitian rho, scaled so that a linear head with weights
    (M_aa, Re M_ab, Im M_ab) reproduces tr(rho M) exactly."""
    n, m, _ = rhos.shape
    iu = np.triu_indices(m, 1)
    return np.concatenate([np.real(np.einsum("naa->na", rhos)),
                           2.0 * np.real(rhos[:, iu[0], iu[1]]),
                           2.0 * np.imag(rhos[:, iu[0], iu[1]])], axis=1)


# ===========================================================================
# classifiers (convex logistic regression; non-convex MLP control)
# ===========================================================================
def _standardize(Xtr, Xte):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def fit_logreg(Xtr, ytr, Xte, yte):
    Xtr, Xte = _standardize(Xtr, Xte)
    n, p = Xtr.shape
    yv = ytr.astype(float)

    def obj(th):
        w, b = th[:p], th[p]
        z = Xtr @ w + b
        loss = float(np.mean(np.logaddexp(0.0, z) - yv * z)) + L2 * float(w @ w)
        s = 1.0 / (1.0 + np.exp(-z))
        gw = Xtr.T @ (s - yv) / n + 2 * L2 * w
        gb = float(np.mean(s - yv))
        return loss, np.concatenate([gw, [gb]])

    r = minimize(obj, np.zeros(p + 1), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=MAXITER, maxfun=10 * MAXITER))
    w, b = r.x[:p], r.x[p]
    acc_tr = float(np.mean(((Xtr @ w + b) > 0) == ytr))
    acc_te = float(np.mean(((Xte @ w + b) > 0) == yte))
    return acc_tr, acc_te, p + 1


def _mlp_unpack(th, p, h):
    i = 0
    W1 = th[i:i + p * h].reshape(p, h); i += p * h
    b1 = th[i:i + h]; i += h
    w2 = th[i:i + h]; i += h
    b2 = th[i]
    return W1, b1, w2, b2


def fit_mlp(Xtr, ytr, Xte, yte, h, seed):
    Xtr, Xte = _standardize(Xtr, Xte)
    n, p = Xtr.shape
    yv = ytr.astype(float)
    nparam = p * h + h + h + 1

    def obj(th):
        W1, b1, w2, b2 = _mlp_unpack(th, p, h)
        A = Xtr @ W1 + b1
        Hh = np.tanh(A)
        z = Hh @ w2 + b2
        loss = float(np.mean(np.logaddexp(0.0, z) - yv * z)) + L2 * (float(W1.ravel() @ W1.ravel()) + float(w2 @ w2))
        s = (1.0 / (1.0 + np.exp(-z)) - yv) / n
        gw2 = Hh.T @ s + 2 * L2 * w2
        gb2 = float(s.sum())
        dH = np.outer(s, w2) * (1 - Hh ** 2)
        gW1 = Xtr.T @ dH + 2 * L2 * W1
        gb1 = dH.sum(0)
        return loss, np.concatenate([gW1.ravel(), gb1, gw2, [gb2]])

    best = None
    for r_i in range(N_RESTARTS):
        rng = np.random.default_rng(10_000 * seed + r_i)
        th0 = rng.normal(scale=0.3, size=nparam)
        r = minimize(obj, th0, jac=True, method="L-BFGS-B",
                     options=dict(maxiter=MAXITER, maxfun=10 * MAXITER))
        if best is None or r.fun < best.fun:          # selected on TRAIN loss only
            best = r
    W1, b1, w2, b2 = _mlp_unpack(best.x, p, h)
    ztr = np.tanh(Xtr @ W1 + b1) @ w2 + b2
    zte = np.tanh(Xte @ W1 + b1) @ w2 + b2
    return float(np.mean((ztr > 0) == ytr)), float(np.mean((zte > 0) == yte)), nparam


# ===========================================================================
# structural checks (these are the primary asserts)
# ===========================================================================
def structure_check(name, rep, T, L, R, F, abelian):
    m = rep.order
    rng = np.random.default_rng(7)
    psi = rng.normal(size=m) + 1j * rng.normal(size=m)
    psi /= np.linalg.norm(psi)
    rho = frame_state(psi, L)

    # (a) rho is a legitimate reduced density matrix
    err_herm = float(np.max(np.abs(rho - rho.conj().T)))
    err_trace = float(abs(np.trace(rho).real - 1.0))
    min_eig = float(np.linalg.eigvalsh(rho).min())

    # (b) rho = (1/|G|) sum_k f(k) R_k  (G-circulant)
    f = np.array([np.vdot(psi, L[k] @ psi) for k in range(m)])
    err_circ = float(np.max(np.abs(rho - np.tensordot(f, R, axes=(0, 0)) / m)))

    # (c) the Fourier basis block-diagonalises rho; is it FULLY diagonal?
    B = F.conj().T @ rho @ F
    offdiag = float(np.max(np.abs(B - np.diag(np.diag(B)))))

    # (d) KEY IDENTITY for the learned arm: unitary readout + linear head == tr(rho M),
    #     M = R diag(w) R^dag Hermitian and arbitrary.
    rr = np.random.default_rng(11)
    Q, _ = np.linalg.qr(rr.normal(size=(m, m)) + 1j * rr.normal(size=(m, m)))
    wv = rr.normal(size=m)
    lhs = float(np.sum(wv * np.real(np.einsum("ak,ab,bk->k", Q.conj(), rho, Q))))
    M = Q @ np.diag(wv.astype(complex)) @ Q.conj().T
    rhs = float(np.real(np.trace(rho @ M)))
    err_identity = abs(lhs - rhs)

    print(f"[struct] {name:>3} |G|={m} abelian={abelian}: "
          f"herm={err_herm:.1e} trace_err={err_trace:.1e} min_eig={min_eig:.1e} "
          f"circulant_err={err_circ:.1e} | max|offdiag of F^dag rho F|={offdiag:.3e} "
          f"| readout==tr(rho M) err={err_identity:.1e}")

    assert err_herm < 1e-12 and err_trace < 1e-12 and min_eig > -1e-12
    assert err_circ < 1e-12
    assert err_identity < 1e-12
    if abelian:
        assert offdiag < 1e-12, "abelian frame state must be Fourier-DIAGONAL (gate theorem)"
    return dict(group=name, order=m, abelian=int(abelian), hermiticity_err=err_herm,
                trace_err=err_trace, min_eigenvalue=min_eig, circulant_err=err_circ,
                max_offdiag_after_fourier=offdiag, readout_eq_trRhoM_err=err_identity)


# ===========================================================================
# one (group, task, seed) run
# ===========================================================================
def make_dataset(m, L, F0, task, seed):
    rng = np.random.default_rng(1000 + seed)
    n = N_TRAIN + N_TEST
    P = rng.normal(size=(n, m)) + 1j * rng.normal(size=(n, m))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    V = np.einsum("gij,nj->ngi", L, P)                       # (n, |G|, |G|) rows L_g psi
    rhos = np.einsum("ngi,nhi->ngh", V, V.conj()) / m        # rho[g,h] = <L_h psi|L_g psi>/|G|

    Hc = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    W = (Hc + Hc.conj().T) / 2.0
    if task == "offdiag_W":
        Mw = F0.conj().T @ W @ F0
        Mw = Mw - np.diag(np.diag(Mw))                       # keep only off-Fourier-diagonal part
        W = F0 @ Mw @ F0.conj().T
        W = (W + W.conj().T) / 2.0
    v = np.real(np.einsum("nab,ba->n", rhos, W))
    thr = float(np.median(v[:N_TRAIN]))                      # threshold from TRAIN only
    y = (v > thr).astype(int)
    # DIAGNOSTIC (must be reported): the physical scale of the label functional. For ABELIAN G
    # the off-Fourier-diagonal part of rho is exactly zero, so the "offdiag_W" label functional
    # is identically 0 and its computed value is pure floating-point roundoff (~1e-16). Any
    # accuracy on such a cell is an artifact of deterministic roundoff, NOT a physical effect.
    sig = float(np.std(v))
    return rhos[:N_TRAIN], y[:N_TRAIN], rhos[N_TRAIN:], y[N_TRAIN:], sig


def run_cell(name, m, L, bases, task, seed):
    F0 = bases[CANONICAL_BASIS]
    Rtr, ytr, Rte, yte, sig = make_dataset(m, L, F0, task, seed)

    # arm 1: fixed Mode F (canonical basis) + linear head
    a_tr, acc_modeF, p_modeF = fit_logreg(feat_modeF(Rtr, F0), ytr, feat_modeF(Rte, F0), yte)

    # arm 1b: best of N_FIXED_BASES fixed Fourier bases, selected on TRAIN accuracy
    best_tr, best_te = -1.0, np.nan
    for F in bases:
        t_tr, t_te, _ = fit_logreg(feat_modeF(Rtr, F), ytr, feat_modeF(Rte, F), yte)
        if t_tr > best_tr:
            best_tr, best_te = t_tr, t_te
    acc_modeF_bestbasis = best_te

    # arm 2: parameter-matched control -- Mode F information, |G|^2-ish trainable params
    h = max(1, int(round(m * m / (m + 2))))
    _, acc_modeF_head, p_head = fit_mlp(feat_modeF(Rtr, F0), ytr, feat_modeF(Rte, F0), yte, h, seed)

    # arm 3: LEARNED readout + linear head  ==  convex logistic regression on tr(rho M)
    _, acc_learned, p_learned = fit_logreg(feat_full(Rtr), ytr, feat_full(Rte), yte)

    return dict(group=name, task=task, seed=seed,
                acc_modeF=acc_modeF, acc_modeF_plus_head=acc_modeF_head,
                acc_learned=acc_learned, acc_modeF_bestbasis=acc_modeF_bestbasis,
                delta=acc_learned - acc_modeF,
                n_params_modeF=p_modeF, n_params_modeF_plus_head=p_head,
                n_params_learned=p_learned, label_signal_scale=sig,
                label_is_roundoff=int(sig < 1e-10),
                majority_baseline=float(max(yte.mean(), 1 - yte.mean())))


# ===========================================================================
# driver
# ===========================================================================
def main():
    os.makedirs(OUT, exist_ok=True)
    groups = [("C6", cyclic(6)), ("S3", symmetric(3)), ("C8", cyclic(8)), ("D4", dihedral4())]
    tasks = ["generic_W", "offdiag_W"]

    struct_rows, rows = [], []
    for name, rep in groups:
        T = mult_table(rep)
        L, R = regular_ops(T)
        ab = is_abelian(T)
        bases = [fourier_basis(L, R, b) for b in range(N_FIXED_BASES)]
        struct_rows.append(structure_check(name, rep, T, L, R, bases[CANONICAL_BASIS], ab))
        for task in tasks:
            for sd in SEEDS:
                r = run_cell(name, rep.order, L, bases, task, sd)
                r["abelian"] = int(ab)
                rows.append(r)
            d = np.array([x["delta"] for x in rows[-len(SEEDS):]])
            se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
            for x in rows[-len(SEEDS):]:
                x["se"] = se
            mf = np.mean([x["acc_modeF"] for x in rows[-len(SEEDS):]])
            mh = np.mean([x["acc_modeF_plus_head"] for x in rows[-len(SEEDS):]])
            ml = np.mean([x["acc_learned"] for x in rows[-len(SEEDS):]])
            mb = np.mean([x["acc_modeF_bestbasis"] for x in rows[-len(SEEDS):]])
            sig = np.mean([x["label_signal_scale"] for x in rows[-len(SEEDS):]])
            flag = "  [VACUOUS: label functional is identically 0, value is roundoff]" if sig < 1e-10 else ""
            vrd = ("vacuous_excluded" if sig < 1e-10
                   else ("SURVIVES" if d.mean() > 2 * se else "negative"))
            print(f"  {name:>3} abelian={int(ab)} {task:>10} | modeF={mf:.5f} "
                  f"modeF_bestbasis={mb:.5f} modeF+head={mh:.5f} learned={ml:.5f} "
                  f"| delta={d.mean():+.5f} se={se:.5f} 2se={2*se:.5f} -> "
                  f"{vrd} | label_scale={sig:.2e}{flag}")

    cols = ["group", "abelian", "task", "seed", "acc_modeF", "acc_modeF_plus_head",
            "acc_learned", "acc_modeF_bestbasis", "delta", "se", "majority_baseline",
            "label_signal_scale", "label_is_roundoff",
            "n_params_modeF", "n_params_modeF_plus_head", "n_params_learned"]
    p1 = os.path.join(OUT, "t2_nonabelian_learned_readout.csv")
    with open(p1, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    # summary + pre-registered verdict
    srows = []
    for name, rep in groups:
        for task in tasks:
            sub = [x for x in rows if x["group"] == name and x["task"] == task]
            d = np.array([x["delta"] for x in sub])
            se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
            sig = float(np.mean([x["label_signal_scale"] for x in sub]))
            vacuous = sig < 1e-10
            srows.append(dict(
                group=name, abelian=sub[0]["abelian"], task=task, n_seeds=len(sub),
                mean_acc_modeF=float(np.mean([x["acc_modeF"] for x in sub])),
                mean_acc_modeF_bestbasis=float(np.mean([x["acc_modeF_bestbasis"] for x in sub])),
                mean_acc_modeF_plus_head=float(np.mean([x["acc_modeF_plus_head"] for x in sub])),
                mean_acc_learned=float(np.mean([x["acc_learned"] for x in sub])),
                mean_delta=float(d.mean()), se=se, two_se=2 * se,
                # Vacuity is tested FIRST. With no label signal both arms fit
                # roundoff, so "delta > 2se" would threshold noise against noise.
                verdict=("vacuous_excluded" if vacuous
                         else ("SURVIVES" if d.mean() > 2 * se else "negative")),
                label_signal_scale=sig,
                cell_is_vacuous=int(vacuous),
                gain_from_params=float(np.mean([x["acc_modeF_plus_head"] for x in sub])
                                       - np.mean([x["acc_modeF"] for x in sub])),
                gain_from_mechanism=float(np.mean([x["acc_learned"] for x in sub])
                                          - np.mean([x["acc_modeF_plus_head"] for x in sub]))))
    p2 = os.path.join(OUT, "t2_nonabelian_summary.csv")
    with open(p2, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0].keys()))
        w.writeheader(); w.writerows(srows)

    p3 = os.path.join(OUT, "t2_structure_check.csv")
    with open(p3, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(struct_rows[0].keys()))
        w.writeheader(); w.writerows(struct_rows)

    print("\nwrote", p1, "\nwrote", p2, "\nwrote", p3)


if __name__ == "__main__":
    main()
