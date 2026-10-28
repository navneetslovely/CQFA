"""
U1 -- SHOT-MATCHED non-abelian learned readout vs fixed Mode F, on a group ladder.
=================================================================================

WHY THIS FILE EXISTS
--------------------
T2 (experiments/readout_basis/t2_nonabelian_learned_readout.py,
results/readout_basis/t2_nonabelian_summary.csv) found that on non-abelian G a LEARNED frame
readout beats the fixed Fourier (Mode F) readout by +0.218 (S3) and +0.184 (D4) on the
generic_W task. An adversarial verifier accepted it but recorded two scope limits:

  (a) T2's "learned" arm was convex logistic regression on ALL |G|^2 Hermitian degrees of
      freedom of the frame state. That is a TOMOGRAPHICALLY COMPLETE readout, statevector
      exact, no shot noise, no circuit. It proves the information EXISTS off the Fourier
      diagonal. It does NOT exhibit a shot-efficient physical protocol.
  (b) Only two non-abelian groups, both of order <= 8. No scaling claim.

Adding groups to the tomographic version makes (a) worse, not better: tomography of |G|^2
numbers costs ~|G|^2/eps^2 shots, so a growing gap would just be measuring a growing
tomography budget. The only version worth running is SHOT-MATCHED, which is this file.

PHYSICAL SETUP (identical to T1's readout family, extended to non-abelian G)
---------------------------------------------------------------------------
Frame register: n = ceil(log2|G|) qubits, D = 2^n.  |Psi> = |G|^{-1/2} sum_g |g> (x) U_g|psi>,
U = LEFT-REGULAR representation.  The frame reduced state is the Gram matrix

        rho[g,h] = (1/|G|) <U_h psi | U_g psi>                          (G-circulant)

embedded in the top-left |G| x |G| block of a D x D matrix (padding states carry zero
amplitude).  A readout is ONE unitary R on the frame register followed by a COMPUTATIONAL
BASIS measurement, giving D outcome probabilities

        p_j(R) = (R^dag rho R)_{jj},      sum_j p_j = 1.

Every arm sees exactly this, and nothing else: no arm ever gets a number that is not an
estimate of some p_j from a finite number of computational-basis samples. That is what
closes T2 scope limit (a).

  * R = F_G (the GENERALIZED FOURIER TRANSFORM, block-diagonalising C[G] into one
    d_lambda x d_lambda block per irrep) reproduces the Mode-F readout of T2 EXACTLY.
    Asserted at run time to < 1e-14 against T2's own functions for S3 and D4
    (`assert_matches_t2`), and F_G is verified to be a genuine GFT by an independent
    irrep-structure check (`gft_structure_check`): the isotypic components are recovered
    from the CENTRE of the group algebra (class sums), the irrep dimensions d_lambda are
    read off as sqrt(dim of isotypic component), sum_lambda d_lambda^2 == |G| and
    #irreps == #conjugacy classes are asserted, and F_G^dag rho F_G is verified to be
    BLOCK diagonal with exactly that multiset of block sizes. For abelian G all
    d_lambda = 1, so the blocks are 1x1 and the readout is fully diagonal -- the C1--C3
    ceiling theorem.

ARMS  (same train/test split, same data budget, same per-state shot budget S)
----------------------------------------------------------------------------
  (1) modeF_fixed      R = F_G (fixed, data independent) + trainable linear head.
                       params = D + 1.
  (2) learned_unitary  R = expm(i H), H Hermitian, D^2 real parameters, trained END TO END
                       with the head on the TRAIN split under shot noise.
                       params = D^2 + D + 1.
  (3) learned_circuit  R = a SHALLOW HARDWARE-EFFICIENT CIRCUIT on n = ceil(log2|G|) qubits:
                       L_CIRC+1 layers of (RY,RZ) on every qubit separated by a fixed ring
                       of CNOTs. params_R = 2*n*(L_CIRC+1) = O(poly log|G|), NOT a general
                       unitary. Trained end to end with the head under shot noise.
                       THIS IS THE PHYSICALLY IMPLEMENTABLE ARM AND THE ONE THAT MATTERS.
  (4) modeF_plus_head  PARAMETER-MATCHED CONTROL: fixed Mode-F features into a 1-hidden-layer
                       MLP head with ~D^2 parameters, i.e. matched to the LARGER of arms
                       (2)/(3). If a win were merely "more parameters", this arm gets it too.

FAIRNESS (a real objection raised against T2)
---------------------------------------------
  * R is fitted on the TRAIN split only; all reported accuracies are on a held-out TEST split.
  * every arm's head is fitted on the same training data, from the same measurement records.
  * TRAIN accuracy is reported alongside TEST accuracy for every arm. If an arm cannot fit
    its own TRAINING set, that is an information ceiling, and it is the single most
    valuable number in the file.
  * arm (2) is WARM-STARTED at R = F_G on one of its two restarts, so a correctly working
    optimiser can never leave it below Mode F for optimisation reasons alone. Arm (3)
    cannot be warm started (F_G is not in the shallow-circuit family) and uses random
    inits only -- this is a real, disclosed handicap on arm (3).

SHOT ACCOUNTING  (two accountings, both reported; columns in the CSV)
--------------------------------------------------------------------
  ACCOUNTING A -- DEPLOYMENT MATCHED (primary; the pre-registered criterion uses this).
    Every arm measures every train and test state with exactly S shots.
        deploy_shots = S * (N_TRAIN + N_TEST),  IDENTICAL for all four arms, ASSERTED in code.
    The cost of FITTING R is reported separately in `readout_fit_shots` (0 for arms 1 and 4)
    and is NOT charged. This is the usual "training cost is amortised over deployment"
    convention, and it is the reading of "shot matched" that isolates the physics question:
    at equal per-state measurement cost, is there a better readout than Mode F?

  ACCOUNTING B -- TOTAL BUDGET MATCHED (secondary, reported prominently, NOT amortised).
    Arms (2)/(3) really do spend readout_fit_shots = n_restarts * EPOCHS * N_TRAIN * S shots
    re-measuring the train split at every optimiser step (the basis changes each step, so a
    single frozen measurement record is not physically available). Accounting B gives arm (1)
    that ENTIRE budget instead, spent on more shots per state in the fixed Fourier basis:
        S_B = (readout_fit_shots + deploy_shots) / (N_TRAIN + N_TEST)
    and reports `acc_test_modeF_totalmatched`. This answers: given one fixed total number of
    measurements, is it better to buy more shots in a fixed basis, or to buy a better basis?

  For the "exact" budget the shot columns are left EMPTY (statevector, no shots) rather than
  being filled with a sentinel, per the censoring rule.

TASKS  (identical functional form for abelian and non-abelian G, so the ladder is clean)
----------------------------------------------------------------------------------------
  psi Haar-random in C^{|G|};  y = 1{ Re tr(rho W) > median_train } for per-seed Hermitian W.
    generic_W  : W generic Hermitian.                          <-- PRIMARY TASK
    offdiag_W  : W supported purely OFF the Fourier block diagonal, i.e. exactly the part a
                 fixed-basis readout is blind to.               <-- SECONDARY, mechanistic
  For ABELIAN G the off-Fourier-diagonal part of rho is exactly zero, so offdiag_W has an
  identically-zero label functional and is pure floating-point roundoff -- the built-in
  vacuity control.

GROUP LADDER
------------
  non-abelian: S3(6) D4(8) Q8(8) D5(10) A4(12) D6(12) D10(20) F21=C7:C3(21) S4(24)
  abelian NEGATIVE CONTROLS: C6(6) C8(8) C12(12)   -- these MUST show no gain.
  Per group we also record max|offdiag of F_G^dag rho F_G|, the residual off-Fourier-diagonal
  weight. The scaling story we would LIKE is that the gap tracks this quantity. We TEST that
  (Pearson r across non-abelian groups, reported); we do not assume it.

=========================================================================================
PRE-REGISTERED CRITERION  -- written before any run; applied exactly as written.
=========================================================================================
Per-cell (group, task, S) statistics over SEEDS (5 seeds, project minimum):
    delta_circ = acc_test(learned_circuit) - acc_test(modeF_fixed)      [ACCOUNTING A]
    se         = std(delta_circ, ddof=1) / sqrt(n_seeds)

VACUITY GUARD, APPLIED FIRST, BEFORE ANY DELTA TEST:
    if label_signal_scale < 1e-10 the cell is `vacuous_excluded` regardless of delta.
    (Without this guard the test thresholds floating-point roundoff against roundoff.)

Per-cell verdict:
    vacuous_excluded   if label_signal_scale < 1e-10
    CONFIRMS_CELL      if mean(delta_circ) > 2*se
    KILLS_CELL         otherwise

TRACK-LEVEL VERDICT, decided at the largest finite shot budget S = 256 on the PRIMARY task
generic_W (this budget/task pair is fixed here, before running):
    CONFIRMS   iff  >= 5 of the 9 non-abelian groups are CONFIRMS_CELL
                AND 0 of the 3 abelian controls are CONFIRMS_CELL.
    KILLS      iff  <  5 of the 9 non-abelian groups are CONFIRMS_CELL.
                In that case the T2 advantage did NOT survive into a physically realisable,
                shot-matched protocol. That is a genuinely important NEGATIVE result and it
                MUST be reported as the headline, plainly, not buried.
    MIXED      iff  the non-abelian majority confirms BUT at least one abelian control also
                confirms. An abelian gain at matched shots is a RED FLAG ABOUT THE HARNESS,
                not a discovery: the C1--C3 ceiling theorem forbids it. Investigate the
                harness before reporting anything.

Secondary, also pre-registered, reported but NOT used to decide the track verdict:
    * the same test with arm (2) learned_unitary in place of arm (3);
    * the same test under ACCOUNTING B;
    * the offdiag_W task;
    * Pearson r between mean(delta_circ) and max_offdiag_after_fourier over non-abelian groups.

CSV OUTPUT
----------
  results/readout_basis/u1_shotmatched_percell.csv     per (group, task, S, seed)
  results/readout_basis/u1_shotmatched_summary.csv     per (group, task, S) with verdicts
  results/readout_basis/u1_structure_check.csv         GFT / irrep / T2-match structural asserts
  results/readout_basis/u1_verdict.csv                 the track-level verdict, one row
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys
import time

import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import minimize

_ROOT = os.path.abspath(__file__)
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(
        os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

OUT = os.path.join(_ROOT, "results", "readout_basis")

# ============================================================================
# PRE-REGISTERED hyper-parameters. Fixed BEFORE the first comparison run.
# ============================================================================
SEEDS = [0, 1, 2, 3, 4]
N_TRAIN = 400
N_TEST = 400
L2 = 1e-3
SHOT_GRID = ["16", "64", "256", "exact"]
DECIDING_SHOTS = 256          # the budget at which the track verdict is read
DECIDING_TASK = "generic_W"   # the task at which the track verdict is read
NONAB_MAJORITY = 5            # >= 5 of 9 non-abelian groups must confirm

L_CIRC = 3                    # hardware-efficient ansatz depth (L_CIRC+1 rotation layers)
EPOCHS = 150                  # Adam epochs per restart for the learned readouts
N_RESTARTS_R = 2              # restart 0 = warm start at F_G (arm 2 only), restart 1 = random
LR_WARM = 0.02
LR_RAND = 0.05
MLP_RESTARTS = 3              # for the parameter-matched MLP control head
MAXITER = 2000                # L-BFGS budget for the (convex) heads
BASIS_SEED = 12345            # identical to T2; makes F_G data independent AND T2 identical
N_FIXED_BASES = 8             # candidate fixed Fourier bases for the best-basis diagnostic
CANONICAL_BASIS = 0
VACUITY_TOL = 1e-10
S_B_CAP = 2_000_000           # cap on the Accounting-B per-state shot count (disclosed)


# ============================================================================
# GROUPS -- explicit multiplication tables
# ============================================================================
def _table(elems, mul):
    idx = {e: i for i, e in enumerate(elems)}
    m = len(elems)
    T = np.zeros((m, m), dtype=int)
    for a, ea in enumerate(elems):
        for b, eb in enumerate(elems):
            T[a, b] = idx[mul(ea, eb)]
    return T


def _check_group(T):
    """identity, closure, associativity, inverses -- asserted, not assumed."""
    m = T.shape[0]
    assert T.shape == (m, m) and set(np.unique(T)) == set(range(m))
    e = [i for i in range(m) if all(T[i, b] == b for b in range(m))
         and all(T[a, i] == a for a in range(m))]
    assert len(e) == 1, "no unique identity"
    for a in range(m):
        assert len(set(T[a, :])) == m and len(set(T[:, a])) == m, "not a Latin square"
    rng = np.random.default_rng(0)
    for _ in range(400):
        a, b, c = rng.integers(0, m, 3)
        assert T[T[a, b], c] == T[a, T[b, c]], "not associative"
    return e[0]


def _perm_mul(p, q):
    return tuple(p[q[i]] for i in range(len(p)))


def group_table(name):
    """Return (mult_table, order). Element ordering for S3 and D4 is taken from cqfa.group
    via T2's own mult_table so that F_G is BIT-IDENTICAL to T2's canonical Fourier basis."""
    if name in ("S3", "D4"):
        from cqfa.group import symmetric, dihedral4
        from experiments.readout_basis.t2_nonabelian_learned_readout import mult_table as t2_mt
        rep = symmetric(3) if name == "S3" else dihedral4()
        T = t2_mt(rep)
    elif name.startswith("C") and name[1:].isdigit():
        n = int(name[1:])
        T = _table(list(range(n)), lambda a, b: (a + b) % n)
    elif name.startswith("D") and name[1:].isdigit():
        n = int(name[1:])                       # D_n, order 2n
        el = [(r, s) for s in (0, 1) for r in range(n)]
        T = _table(el, lambda x, y: (((x[0] + (1 - 2 * x[1]) * y[0]) % n), (x[1] + y[1]) % 2))
    elif name == "Q8":                          # dicyclic: x^a y^b, x^4=1, y^2=x^2, yxy^-1=x^-1
        el = [(a, b) for b in (0, 1) for a in range(4)]

        def mul(x, y):
            a1, b1 = x
            a2, b2 = y
            if b1 == 0:
                return ((a1 + a2) % 4, b2)
            t = a1 - a2
            return ((t + (2 if b2 == 1 else 0)) % 4, (1 + b2) % 2)
        T = _table(el, mul)
    elif name == "A4":
        el = [p for p in itertools.permutations(range(4))
              if _parity(p) == 0]
        T = _table(el, _perm_mul)
    elif name == "S4":
        el = list(itertools.permutations(range(4)))
        T = _table(el, _perm_mul)
    elif name == "F21":                         # C7 : C3, action x -> 2x  (2^3 = 1 mod 7)
        el = [(a, b) for b in range(3) for a in range(7)]
        T = _table(el, lambda x, y: ((x[0] + pow(2, x[1], 7) * y[0]) % 7, (x[1] + y[1]) % 3))
    else:
        raise ValueError(name)
    _check_group(T)
    return T, T.shape[0]


def _parity(p):
    n = len(p)
    seen = [False] * n
    par = 0
    for i in range(n):
        if not seen[i]:
            j, ln = i, 0
            while not seen[j]:
                seen[j] = True
                j = p[j]
                ln += 1
            par += ln - 1
    return par % 2


def regular_ops(T):
    """L_a|b> = |ab>, R_a|b> = |ba> on C^{|G|}."""
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


def n_order2(T):
    """number of involutions -- distinguishes Q8 (1) from D4 (5)."""
    m = T.shape[0]
    e = [i for i in range(m) if all(T[i, b] == b for b in range(m))][0]
    return sum(1 for a in range(m) if a != e and T[a, a] == e)


# ============================================================================
# GENERALIZED FOURIER TRANSFORM
# ============================================================================
def fourier_basis(L, R, b_index=CANONICAL_BASIS):
    """The GFT / matrix-unit basis of C[G]. IDENTICAL CODE to T2's fourier_basis.

    H = (A + A^dag)/2 with A = sum_k c_k R_k + sum_k d_k L_k is a sum of a Hermitian element
    of the RIGHT-regular algebra and a Hermitian element of the LEFT-regular algebra; the two
    commute. In C[G] = (+)_lambda M_{d_lambda} the left algebra acts by left multiplication
    and the right algebra by right multiplication, so the joint eigenvectors of H are the
    MATRIX UNITS E^lambda_{ab} in bases that diagonalise the two generic elements. That is
    exactly a choice of generalized Fourier transform. Deterministic in BASIS_SEED.
    """
    m = L.shape[0]
    rng = np.random.default_rng(BASIS_SEED + 1000 * b_index)
    c = rng.normal(size=m) + 1j * rng.normal(size=m)
    d = rng.normal(size=m) + 1j * rng.normal(size=m)
    A = np.tensordot(c, R, axes=(0, 0)) + np.tensordot(d, L, axes=(0, 0))
    H = (A + A.conj().T) / 2.0
    w, F = np.linalg.eigh(H)
    assert np.diff(np.sort(w)).min() > 1e-6, "degenerate spectrum -- basis ill defined"
    return F


def conjugacy_classes(T):
    m = T.shape[0]
    inv = np.zeros(m, dtype=int)
    e = [i for i in range(m) if all(T[i, b] == b for b in range(m))][0]
    for a in range(m):
        inv[a] = int(np.where(T[a, :] == e)[0][0])
    cls, seen = [], set()
    for a in range(m):
        if a in seen:
            continue
        c = {T[T[g, a], inv[g]] for g in range(m)}
        cls.append(sorted(c))
        seen |= c
    return cls


def irrep_dims_from_centre(T, R):
    """Irrep dimensions read off the CENTRE of the group algebra (class sums).

    The class sums span the centre; a generic Hermitian element of the centre has the
    ISOTYPIC components of the regular representation as eigenspaces, each of dimension
    d_lambda^2. So d_lambda = sqrt(dim of eigenspace). Asserts sum d^2 == |G| and
    #irreps == #conjugacy classes -- both are theorems, so a failure means a harness bug.
    """
    m = T.shape[0]
    cls = conjugacy_classes(T)
    rng = np.random.default_rng(BASIS_SEED + 7)
    Zc = np.zeros((m, m), dtype=complex)
    for i, c in enumerate(cls):
        Cs = sum(R[k] for k in c)
        Zc = Zc + rng.normal() * (Cs + Cs.conj().T) / 2.0
    w, V = np.linalg.eigh(Zc)
    # cluster eigenvalues
    groups, cur = [], [0]
    for i in range(1, m):
        if w[i] - w[i - 1] > 1e-7 * max(1.0, abs(w[i])):
            groups.append(cur); cur = []
        cur.append(i)
    groups.append(cur)
    dims = [len(g) for g in groups]
    ds = [int(round(math.sqrt(d))) for d in dims]
    assert all(abs(d - dd * dd) < 1e-9 for d, dd in zip(dims, ds)), \
        f"isotypic dims {dims} are not perfect squares -- centre split failed"
    assert sum(dd * dd for dd in ds) == m, "sum d_lambda^2 != |G|"
    assert len(ds) == len(cls), "#irreps != #conjugacy classes"
    projs = [V[:, g] @ V[:, g].conj().T for g in groups]
    return sorted(ds), projs


def gft_block_structure(F, rho_list, tol=1e-9):
    """Block sizes of F^dag rho F, found as connected components of the |B| > tol graph,
    accumulated over several rho so an accidental zero cannot merge blocks."""
    D = F.shape[0]
    A = np.zeros((D, D), dtype=bool)
    for rho in rho_list:
        B = np.abs(F.conj().T @ rho @ F)
        A |= (B > tol)
    A |= A.T
    seen, sizes = set(), []
    for i in range(D):
        if i in seen:
            continue
        stack, comp = [i], set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack.extend(int(v) for v in np.where(A[u])[0] if v not in comp)
        seen |= comp
        sizes.append(len(comp))
    return sorted(sizes)


# ============================================================================
# frame state
# ============================================================================
def frame_states(P, L):
    """rho_n[g,h] = (1/|G|) <L_h psi_n | L_g psi_n>."""
    m = L.shape[0]
    V = np.einsum("gij,nj->ngi", L, P)
    return np.einsum("ngi,nhi->ngh", V, V.conj()) / m


# ---- real vectorisation of Hermitian matrices: <phi(rho), phi(M)> = tr(rho M) ----
def _pair_idx(m):
    return np.triu_indices(m, 1)


def phi(rhos, iu):
    """(n,m,m) Hermitian -> (n, m^2) real, with <phi(a),phi(b)> = tr(a b)."""
    d = np.real(np.einsum("naa->na", rhos))
    off = rhos[:, iu[0], iu[1]]
    s = math.sqrt(2.0)
    return np.concatenate([d, s * np.real(off), s * np.imag(off)], axis=1)


def unphi(v, m, iu):
    """(k, m^2) real -> (k, m, m) Hermitian; inverse of phi."""
    k = v.shape[0]
    npair = len(iu[0])
    M = np.zeros((k, m, m), dtype=complex)
    M[:, np.arange(m), np.arange(m)] = v[:, :m]
    s = math.sqrt(2.0)
    off = (v[:, m:m + npair] + 1j * v[:, m + npair:]) / s
    M[:, iu[0], iu[1]] = off
    M[:, iu[1], iu[0]] = off.conj()
    return M


# ============================================================================
# readout parameterisations
# ============================================================================
def herm_from_theta(th, D):
    iu = np.triu_indices(D, 1)
    H = np.zeros((D, D), dtype=complex)
    H[np.diag_indices(D)] = th[:D]
    nod = len(iu[0])
    H[iu] = th[D:D + nod] + 1j * th[D + nod:D + 2 * nod]
    return H + H.conj().T - np.diag(np.diag(H).real)


def theta_from_herm(H, D):
    iu = np.triu_indices(D, 1)
    return np.concatenate([np.real(np.diag(H)), np.real(H[iu]), np.imag(H[iu])])


def unitary_from_theta(th, D):
    return expm(1j * herm_from_theta(th, D))


def grad_theta_from_dR(Z, th, D):
    """Given Z = dL/d conj(R)  (so dL = 2 Re tr(Z^dag dR)), return dL/dtheta for R=expm(iH)."""
    H = herm_from_theta(th, D)
    lam, V = np.linalg.eigh(H)
    el = np.exp(1j * lam)
    dl = lam[:, None] - lam[None, :]
    Phi = np.where(np.abs(dl) < 1e-9,
                   1j * el[:, None],
                   (el[:, None] - el[None, :]) / np.where(np.abs(dl) < 1e-9, 1.0, dl))
    Zt = V.conj().T @ Z @ V
    Y = np.conj(Phi) * Zt
    Gh = V @ Y @ V.conj().T
    Gh = (Gh + Gh.conj().T) / 2.0            # dL = 2 tr(Gh X) for Hermitian X = dH
    iu = np.triu_indices(D, 1)
    return np.concatenate([2.0 * np.real(np.diag(Gh)),
                           4.0 * np.real(Gh[iu]),
                           4.0 * np.imag(Gh[iu])])


# ---- shallow hardware-efficient circuit ----
_PAULI = {"Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
          "Z": np.array([[1, 0], [0, -1]], dtype=complex)}


def _apply_1q(M, P, q, n):
    """P acting on qubit q (qubit 0 = most significant) applied on the LEFT of M."""
    D = 1 << n
    A = M.reshape(1 << q, 2, 1 << (n - q - 1), D)
    return np.einsum("xy,ayzd->axzd", P, A).reshape(D, D)


def _rot(P, ang, q, n):
    D = 1 << n
    G = np.eye(D, dtype=complex) * math.cos(ang / 2)
    return G + (-1j * math.sin(ang / 2)) * _apply_1q(np.eye(D, dtype=complex), P, q, n)


def _cnot_ring(n):
    D = 1 << n
    U = np.eye(D, dtype=complex)
    if n < 2:
        return U
    for c in range(n):
        t = (c + 1) % n
        G = np.zeros((D, D), dtype=complex)
        for b in range(D):
            bc = (b >> (n - 1 - c)) & 1
            bb = b ^ ((1 << (n - 1 - t)) if bc else 0)
            G[bb, b] = 1.0
        U = G @ U
        if n == 2:
            break
    return U


def circuit_spec(n):
    """List of ('rot', pauli, qubit) and ('fix', matrix); R = prod (last..first)."""
    spec = []
    ring = _cnot_ring(n)
    for lay in range(L_CIRC + 1):
        for q in range(n):
            spec.append(("rot", "Y", q))
            spec.append(("rot", "Z", q))
        if lay < L_CIRC:
            spec.append(("fix", ring))
    return spec


def circuit_nparams(n):
    return 2 * n * (L_CIRC + 1)


def circuit_build(spec, th, n):
    """Return (R, cumulative products Cum[k] = A_k...A_1 for parametric gates)."""
    D = 1 << n
    Cur = np.eye(D, dtype=complex)
    cums, paulis = [], []
    i = 0
    for item in spec:
        if item[0] == "rot":
            _, pl, q = item
            A = _rot(_PAULI[pl], th[i], q, n)
            Cur = A @ Cur
            cums.append(Cur.copy())
            paulis.append((pl, q))
            i += 1
        else:
            Cur = item[1] @ Cur
    return Cur, cums, paulis


def grad_theta_circuit(Z, R, cums, paulis, n):
    """dL/dtheta_k = Re[-i tr(M Cum_k^dag P_k Cum_k)] with M = Z^dag R."""
    M = Z.conj().T @ R
    g = np.zeros(len(cums))
    for k, (Ck, (pl, q)) in enumerate(zip(cums, paulis)):
        PC = _apply_1q(Ck, _PAULI[pl], q, n)
        W = Ck.conj().T @ PC
        g[k] = float(np.real(-1j * np.trace(M @ W)))
    return g


# ============================================================================
# forward model: probabilities and shot sampling
# ============================================================================
def probs(Psi, R, m, iu):
    """p[n,j] = (R^dag rho_n R)_jj, computed as a real GEMM against phi(rho_n)."""
    Rm = R[:m, :]                                   # rho is supported on the first m rows
    # M_j[a,b] = R[a,j] conj(R[b,j]), so tr(rho M_j) = (R^dag rho R)_jj.
    # NOTE the operand order: ("aj,bj->jab", Rm.conj(), Rm) instead evaluates
    # (R^dag rho^T R)_jj -- the TRANSPOSED state. rho^T is still a valid density
    # matrix, so that error is silent here (outcomes stay positive and sum to 1)
    # and only surfaces as a gradient mismatch, because dprobs_to_Z below
    # correctly differentiates the untransposed model. That inconsistency made
    # --selftest fail at rel err 6.7e-01 (unitary) / 4.0e-01 (circuit).
    Mj = np.einsum("aj,bj->jab", Rm, Rm.conj())     # M_j Hermitian
    PhiM = phi(Mj, iu)
    return Psi @ PhiM.T


def dprobs_to_Z(Gp, Psi, R, m, iu):
    """Given Gp[n,j] = dL/dp[n,j], return Z = dL/d conj(R)  (D x D)."""
    D = R.shape[0]
    PhiQ = Gp.T @ Psi                               # (D, m^2)
    Q = unphi(PhiQ, m, iu)                          # Q_j = sum_n Gp[n,j] rho_n
    Z = np.zeros((D, D), dtype=complex)
    Z[:m, :] = np.einsum("jab,bj->aj", Q, R[:m, :])
    return Z


def sample_shots(p, S, rng):
    if S is None:
        return p
    q = np.clip(p, 0.0, None)
    q = q / q.sum(axis=1, keepdims=True)
    return rng.multinomial(S, q) / float(S)


# ============================================================================
# heads
# ============================================================================
def _std(Xtr, Xte):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def fit_logreg(Xtr, ytr, Xte, yte):
    Xtr, Xte = _std(Xtr, Xte)
    n, p = Xtr.shape
    yv = ytr.astype(float)

    def obj(th):
        w, b = th[:p], th[p]
        z = Xtr @ w + b
        loss = float(np.mean(np.logaddexp(0.0, z) - yv * z)) + L2 * float(w @ w)
        s = 1.0 / (1.0 + np.exp(-z))
        return loss, np.concatenate([Xtr.T @ (s - yv) / n + 2 * L2 * w, [float(np.mean(s - yv))]])

    r = minimize(obj, np.zeros(p + 1), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=MAXITER, maxfun=10 * MAXITER))
    w, b = r.x[:p], r.x[p]
    return (float(np.mean(((Xtr @ w + b) > 0) == ytr)),
            float(np.mean(((Xte @ w + b) > 0) == yte)), p + 1)


def _mlp_unpack(th, p, h):
    i = 0
    W1 = th[i:i + p * h].reshape(p, h); i += p * h
    b1 = th[i:i + h]; i += h
    w2 = th[i:i + h]; i += h
    return W1, b1, w2, th[i]


def fit_mlp(Xtr, ytr, Xte, yte, h, seed):
    Xtr, Xte = _std(Xtr, Xte)
    n, p = Xtr.shape
    yv = ytr.astype(float)
    nparam = p * h + 2 * h + 1

    def obj(th):
        W1, b1, w2, b2 = _mlp_unpack(th, p, h)
        Hh = np.tanh(Xtr @ W1 + b1)
        z = Hh @ w2 + b2
        loss = float(np.mean(np.logaddexp(0.0, z) - yv * z)) + \
            L2 * (float(W1.ravel() @ W1.ravel()) + float(w2 @ w2))
        s = (1.0 / (1.0 + np.exp(-z)) - yv) / n
        dH = np.outer(s, w2) * (1 - Hh ** 2)
        return loss, np.concatenate([(Xtr.T @ dH + 2 * L2 * W1).ravel(), dH.sum(0),
                                     Hh.T @ s + 2 * L2 * w2, [float(s.sum())]])

    best = None
    for r_i in range(MLP_RESTARTS):
        rng = np.random.default_rng(10_000 * seed + r_i)
        r = minimize(obj, rng.normal(scale=0.3, size=nparam), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=MAXITER, maxfun=10 * MAXITER))
        if best is None or r.fun < best.fun:
            best = r
    W1, b1, w2, b2 = _mlp_unpack(best.x, p, h)
    ztr = np.tanh(Xtr @ W1 + b1) @ w2 + b2
    zte = np.tanh(Xte @ W1 + b1) @ w2 + b2
    return (float(np.mean((ztr > 0) == ytr)), float(np.mean((zte > 0) == yte)), nparam)


# ============================================================================
# end-to-end training of a readout + linear head under shot noise
# ============================================================================
def train_readout(kind, Psi_tr, ytr, m, D, n, S, seed, F_embed, iu):
    """Adam on (readout params, w, b). At every epoch the TRAIN split is RE-MEASURED with S
    shots in the CURRENT basis -- the physically  cost, counted in shots_used.

    Gradient through the shot noise uses the straight-through estimator
    f = p + (p_hat - p)_detached, i.e. dL/dp = dL/df. The gradient of p w.r.t. the readout
    parameters is EXACT (Daleckii-Krein for expm, gate-by-gate for the circuit), and is
    verified against finite differences in --selftest.
    """
    rng = np.random.default_rng(900_000 + 131 * seed + 7 * D)
    Ntr = Psi_tr.shape[0]
    yv = ytr.astype(float)
    shots_used = 0
    best = None

    for restart in range(N_RESTARTS_R):
        warm = (restart == 0 and kind == "unitary")
        if kind == "unitary":
            npR = D * D
            if warm:
                H0 = -1j * logm(F_embed)
                H0 = (H0 + H0.conj().T) / 2.0
                thR = theta_from_herm(H0, D)
                lr = LR_WARM
            else:
                thR = 0.6 * rng.normal(size=npR)
                lr = LR_RAND
        else:
            spec = circuit_spec(n)
            npR = circuit_nparams(n)
            thR = rng.uniform(-math.pi, math.pi, size=npR)
            lr = LR_RAND

        w = np.zeros(D)
        b = 0.0
        th = np.concatenate([thR, w, [b]])
        mt = np.zeros_like(th)
        vt = np.zeros_like(th)

        for ep in range(EPOCHS):
            if kind == "unitary":
                R = unitary_from_theta(th[:npR], D)
            else:
                R, cums, paulis = circuit_build(spec, th[:npR], n)
            p = probs(Psi_tr, R, m, iu)
            f = sample_shots(p, S, rng)
            if S is not None:
                shots_used += S * Ntr
            mu, sd = f.mean(0), np.where(f.std(0) < 1e-12, 1.0, f.std(0))
            X = (f - mu) / sd
            wv, bv = th[npR:npR + D], th[npR + D]
            z = X @ wv + bv
            s = 1.0 / (1.0 + np.exp(-z))
            gz = (s - yv) / Ntr
            gw = X.T @ gz + 2 * L2 * wv
            gb = float(gz.sum())
            Gp = np.outer(gz, wv) / sd                       # dL/dp via straight-through
            Z = dprobs_to_Z(Gp, Psi_tr, R, m, iu)
            gR = (grad_theta_from_dR(Z, th[:npR], D) if kind == "unitary"
                  else grad_theta_circuit(Z, R, cums, paulis, n))
            g = np.concatenate([gR, gw, [gb]])
            lr_t = lr * 0.5 * (1.0 + math.cos(math.pi * ep / EPOCHS))   # cosine decay to 0
            mt = 0.9 * mt + 0.1 * g
            vt = 0.999 * vt + 0.001 * g * g
            mh = mt / (1 - 0.9 ** (ep + 1))
            vh = vt / (1 - 0.999 ** (ep + 1))
            th = th - lr_t * mh / (np.sqrt(vh) + 1e-8)

        # select the restart on TRAIN loss from ONE extra measurement round (counted)
        if kind == "unitary":
            R = unitary_from_theta(th[:npR], D)
        else:
            R, _, _ = circuit_build(spec, th[:npR], n)
        p = probs(Psi_tr, R, m, iu)
        f = sample_shots(p, S, rng)
        if S is not None:
            shots_used += S * Ntr
        mu, sd = f.mean(0), np.where(f.std(0) < 1e-12, 1.0, f.std(0))
        z = ((f - mu) / sd) @ th[npR:npR + D] + th[npR + D]
        loss = float(np.mean(np.logaddexp(0.0, z) - yv * z))
        if best is None or loss < best[0]:
            best = (loss, R)
    return best[1], shots_used


# ============================================================================
# dataset
# ============================================================================
def make_dataset(m, L, F0, task, seed):
    rng = np.random.default_rng(1000 + seed)
    nn = N_TRAIN + N_TEST
    P = rng.normal(size=(nn, m)) + 1j * rng.normal(size=(nn, m))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    rhos = frame_states(P, L)
    Hc = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    W = (Hc + Hc.conj().T) / 2.0
    if task == "offdiag_W":
        Mw = F0.conj().T @ W @ F0
        Mw = Mw - np.diag(np.diag(Mw))
        W = F0 @ Mw @ F0.conj().T
        W = (W + W.conj().T) / 2.0
    v = np.real(np.einsum("nab,ba->n", rhos, W))
    thr = float(np.median(v[:N_TRAIN]))
    y = (v > thr).astype(int)
    return rhos, y, float(np.std(v))


# ============================================================================
# structural checks (load bearing)
# ============================================================================
def assert_matches_t2(name, m, L, F_m):
    """R = F_G must reproduce T2's Mode-F readout to ~1e-14 on the groups T2 ran."""
    if name not in ("S3", "D4", "C6", "C8"):
        return None
    from experiments.readout_basis.t2_nonabelian_learned_readout import (
        feat_modeF as t2_feat, frame_state as t2_frame)
    rng = np.random.default_rng(4242)
    err = 0.0
    for _ in range(5):
        psi = rng.normal(size=m) + 1j * rng.normal(size=m)
        psi /= np.linalg.norm(psi)
        rho = t2_frame(psi, L)
        ours = np.real(np.einsum("ak,ab,bk->k", F_m.conj(), rho, F_m))
        theirs = t2_feat(rho[None], F_m)[0]
        err = max(err, float(np.max(np.abs(ours - theirs))))
    assert err < 1e-14, f"{name}: readout does not reproduce T2 Mode F (err={err:.2e})"
    return err


def structure_row(name, T, L, R, F_m, abelian):
    m = T.shape[0]
    rng = np.random.default_rng(7)
    rhos = []
    for _ in range(4):
        psi = rng.normal(size=m) + 1j * rng.normal(size=m)
        psi /= np.linalg.norm(psi)
        rhos.append(frame_states(psi[None], L)[0])
    rho = rhos[0]
    err_herm = float(np.max(np.abs(rho - rho.conj().T)))
    err_tr = float(abs(np.trace(rho).real - 1.0))
    min_eig = float(np.linalg.eigvalsh(rho).min())
    f = np.array([np.vdot(psi, L[k] @ psi) for k in range(m)])
    err_circ = float(np.max(np.abs(rhos[-1] - np.tensordot(f, R, axes=(0, 0)) / m)))
    B = F_m.conj().T @ rho @ F_m
    offdiag = float(np.max(np.abs(B - np.diag(np.diag(B)))))
    # The irrep-dimension diagnostic below splits the group algebra centre by
    # eigenvalue clustering. That fails for groups with COMPLEX (non-real) irreps
    # -- A4 has three 1-dimensional irreps, two of them a complex-conjugate pair
    # with cube-root-of-unity characters, which cluster together and are reported
    # as a single 2-dimensional block: isotypic dims come out [1,2,9] instead of
    # [1,1,1,3], and the perfect-square assertion inside irrep_dims_from_centre
    # trips. F21 = C7:C3 has the same issue.
    #
    # This is a DIAGNOSTIC ONLY. irrep_dims_from_centre is used nowhere on the
    # measurement path -- run_cell never calls it -- so a failure here says
    # nothing about the accuracies. Aborting the whole sweep over an unavailable
    # diagnostic would lose the measurement for every later group, so instead we
    # record that it was unavailable and carry on. gft_verified=0 marks the
    # groups for which F_G was NOT independently confirmed to be a generalized
    # Fourier transform by this route; do not claim that verification for them.
    try:
        dims, _ = irrep_dims_from_centre(T, R)
        blocks = gft_block_structure(F_m, rhos)
        expect = sorted([d for d in dims for _ in range(d)])
        gft_verified = int(blocks == expect)
        gft_note = "" if gft_verified else f"block mismatch {blocks} != {expect}"
    except Exception as exc:                      # noqa: BLE001 - recorded, not swallowed
        dims, blocks, expect = [], [], []
        gft_verified = 0
        gft_note = f"centre split unavailable: {exc}"
    t2err = assert_matches_t2(name, m, L, F_m)

    # Physics assertions below are NOT relaxed: state validity, G-circulance and
    # the abelian Fourier-diagonal ceiling must all still hold exactly.
    assert err_herm < 1e-12 and err_tr < 1e-12 and min_eig > -1e-12
    assert err_circ < 1e-12, f"{name}: frame state not G-circulant"
    if gft_note.startswith("block mismatch"):
        raise AssertionError(f"{name}: GFT block sizes {blocks} != irrep multiset {expect} "
                             "-- F_G is not a generalized Fourier transform")
    if abelian:
        assert offdiag < 1e-12, "abelian frame state must be Fourier DIAGONAL (C1-C3 ceiling)"
    return dict(group=name, order=m, abelian=int(abelian),
                n_frame_qubits=max(1, int(math.ceil(math.log2(m)))),
                irrep_dims="|".join(map(str, dims)) if dims else "unavailable",
                n_conjugacy_classes=len(conjugacy_classes(T)),
                gft_block_sizes="|".join(map(str, blocks)) if blocks else "unavailable",
                gft_block_sizes_expected="|".join(map(str, expect)) if expect else "unavailable",
                gft_verified=gft_verified, gft_note=gft_note,
                hermiticity_err=err_herm, trace_err=err_tr, min_eigenvalue=min_eig,
                circulant_err=err_circ, max_offdiag_after_fourier=offdiag,
                modeF_matches_t2_err=("" if t2err is None else t2err),
                n_involutions=n_order2(T))


# ============================================================================
# one (group, task, S, seed) cell
# ============================================================================
def run_cell(name, T, L, Rop, bases, task, S, seed, abelian):
    m = T.shape[0]
    n = max(1, int(math.ceil(math.log2(m))))
    D = 1 << n
    iu = _pair_idx(m)
    F_m = bases[CANONICAL_BASIS]
    F_embed = np.eye(D, dtype=complex)
    F_embed[:m, :m] = F_m

    rhos, y, sig = make_dataset(m, L, F_m, task, seed)
    Psi = phi(rhos, iu)
    Psi_tr, Psi_te = Psi[:N_TRAIN], Psi[N_TRAIN:]
    ytr, yte = y[:N_TRAIN], y[N_TRAIN:]

    ev = np.random.default_rng(555_000 + 977 * seed + 13 * D + (0 if task == "generic_W" else 1))

    def evaluate(R, Sdep):
        p_tr = probs(Psi_tr, R, m, iu)
        p_te = probs(Psi_te, R, m, iu)
        assert abs(float(p_tr.sum(1).mean()) - 1.0) < 1e-9, "outcome probabilities do not sum to 1"
        return fit_logreg(sample_shots(p_tr, Sdep, ev), ytr, sample_shots(p_te, Sdep, ev), yte)

    out = dict(group=name, abelian=int(abelian), order=m, n_frame_qubits=n, D=D,
               task=task, shots=("exact" if S is None else S), seed=seed,
               label_signal_scale=sig, label_is_roundoff=int(sig < VACUITY_TOL),
               majority_baseline=float(max(yte.mean(), 1 - yte.mean())))

    # ---- arm 1: fixed Mode F + linear head ----
    a_tr, a_te, p1 = evaluate(F_embed, S)
    out.update(acc_train_modeF_fixed=a_tr, acc_test_modeF_fixed=a_te, n_params_modeF_fixed=p1)

    # ---- arm 4: parameter-matched MLP control on Mode-F features ----
    h = max(1, int(round(D * D / (D + 2))))
    p_tr = probs(Psi_tr, F_embed, m, iu)
    p_te = probs(Psi_te, F_embed, m, iu)
    f_tr, f_te = sample_shots(p_tr, S, ev), sample_shots(p_te, S, ev)
    b_tr, b_te, p4 = fit_mlp(f_tr, ytr, f_te, yte, h, seed)
    out.update(acc_train_modeF_plus_head=b_tr, acc_test_modeF_plus_head=b_te,
               n_params_modeF_plus_head=p4)

    # ---- diagnostic: best of N_FIXED_BASES fixed Fourier bases, selected on TRAIN acc ----
    best_tr, best_te = -1.0, float("nan")
    for Fb in bases:
        Fe = np.eye(D, dtype=complex); Fe[:m, :m] = Fb
        t_tr, t_te, _ = evaluate(Fe, S)
        if t_tr > best_tr:
            best_tr, best_te = t_tr, t_te
    out.update(acc_train_modeF_bestbasis=best_tr, acc_test_modeF_bestbasis=best_te)

    # ---- arms 2 and 3: learned readouts ----
    fit_shots_total = 0
    for kind, tag, npR in (("unitary", "learned_unitary", D * D),
                           ("circuit", "learned_circuit", circuit_nparams(n))):
        t0 = time.time()
        Rlearn, fs = train_readout(kind, Psi_tr, ytr, m, D, n, S, seed, F_embed, iu)
        fit_shots_total += fs
        c_tr, c_te, _ = evaluate(Rlearn, S)
        out[f"acc_train_{tag}"] = c_tr
        out[f"acc_test_{tag}"] = c_te
        out[f"n_params_{tag}"] = npR + D + 1
        out[f"readout_fit_shots_{tag}"] = ("" if S is None else fs)
        out[f"secs_{tag}"] = round(time.time() - t0, 2)

    # ---- shot accounting ----
    deploy = ("" if S is None else S * (N_TRAIN + N_TEST))
    out["deploy_shots_per_arm"] = deploy
    out["shots_per_state"] = ("" if S is None else S)
    if S is not None:
        # ASSERT deployment shot equality across all four arms (Accounting A)
        per_arm = [S * (N_TRAIN + N_TEST)] * 4
        assert len(set(per_arm)) == 1, "deployment shot budgets are not matched across arms"
        out["total_shots_modeF_fixed"] = S * (N_TRAIN + N_TEST)
        out["total_shots_modeF_plus_head"] = S * (N_TRAIN + N_TEST)
        out["total_shots_learned_unitary"] = (S * (N_TRAIN + N_TEST)
                                              + int(out["readout_fit_shots_learned_unitary"]))
        out["total_shots_learned_circuit"] = (S * (N_TRAIN + N_TEST)
                                              + int(out["readout_fit_shots_learned_circuit"]))
        # ---- ACCOUNTING B: give Mode F the learned_circuit arm's ENTIRE total budget ----
        S_B = int(out["total_shots_learned_circuit"] // (N_TRAIN + N_TEST))
        capped = S_B > S_B_CAP
        S_use = min(S_B, S_B_CAP)
        tb_tr, tb_te, _ = evaluate(F_embed, S_use)
        out["shots_per_state_totalmatched"] = S_use
        out["acc_train_modeF_totalmatched"] = tb_tr
        out["acc_test_modeF_totalmatched"] = tb_te
        out["totalmatched_shots_capped"] = int(capped)
    else:
        for k in ("total_shots_modeF_fixed", "total_shots_modeF_plus_head",
                  "total_shots_learned_unitary", "total_shots_learned_circuit",
                  "shots_per_state_totalmatched", "totalmatched_shots_capped"):
            out[k] = ""
        out["acc_train_modeF_totalmatched"] = a_tr
        out["acc_test_modeF_totalmatched"] = a_te
        # optimisation-gap diagnostic: at S=exact, arm (2) with a LINEAR head is exactly
        # logistic regression over all Hermitian M = R diag(w) R^dag, a CONVEX problem.
        # Solving it directly bounds how much of any arm-(2) shortfall is the optimiser.
        cv_tr, cv_te, _ = fit_logreg(Psi_tr * math.sqrt(1.0), ytr, Psi_te, yte)
        out["acc_train_learned_unitary_convex"] = cv_tr
        out["acc_test_learned_unitary_convex"] = cv_te

    out["delta_circ"] = out["acc_test_learned_circuit"] - out["acc_test_modeF_fixed"]
    out["delta_unit"] = out["acc_test_learned_unitary"] - out["acc_test_modeF_fixed"]
    out["delta_circ_totalmatched"] = (out["acc_test_learned_circuit"]
                                      - out["acc_test_modeF_totalmatched"])
    return out


# ============================================================================
# gradient self-test
# ============================================================================
def selftest():
    print("[selftest] analytic gradients vs finite differences")
    rng = np.random.default_rng(0)
    for name in ("S3", "D4", "A4"):
        T, m = group_table(name)
        L, Rop = regular_ops(T)
        n = max(1, int(math.ceil(math.log2(m))))
        D = 1 << n
        iu = _pair_idx(m)
        P = rng.normal(size=(12, m)) + 1j * rng.normal(size=(12, m))
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        Psi = phi(frame_states(P, L), iu)
        c = rng.normal(size=D)

        def loss_from_p(p):
            return float(np.sum(np.sin(p) * c))

        def gp(p):
            return np.cos(p) * c

        # unitary parameterisation
        th = 0.5 * rng.normal(size=D * D)
        R = unitary_from_theta(th, D)
        p = probs(Psi, R, m, iu)
        Z = dprobs_to_Z(gp(p), Psi, R, m, iu)
        ga = grad_theta_from_dR(Z, th, D)
        idx = rng.choice(D * D, 6, replace=False)
        gn = []
        for i in idx:
            e = np.zeros(D * D); e[i] = 1e-6
            gn.append((loss_from_p(probs(Psi, unitary_from_theta(th + e, D), m, iu))
                       - loss_from_p(probs(Psi, unitary_from_theta(th - e, D), m, iu))) / 2e-6)
        eu = float(np.max(np.abs(ga[idx] - np.array(gn))) / (np.max(np.abs(gn)) + 1e-12))

        # circuit parameterisation
        spec = circuit_spec(n)
        npc = circuit_nparams(n)
        thc = rng.uniform(-math.pi, math.pi, size=npc)
        Rc, cums, paulis = circuit_build(spec, thc, n)
        assert np.max(np.abs(Rc.conj().T @ Rc - np.eye(D))) < 1e-10, "circuit not unitary"
        pc = probs(Psi, Rc, m, iu)
        Zc = dprobs_to_Z(gp(pc), Psi, Rc, m, iu)
        gac = grad_theta_circuit(Zc, Rc, cums, paulis, n)
        gnc = []
        for i in range(npc):
            e = np.zeros(npc); e[i] = 1e-6
            gnc.append((loss_from_p(probs(Psi, circuit_build(spec, thc + e, n)[0], m, iu))
                        - loss_from_p(probs(Psi, circuit_build(spec, thc - e, n)[0], m, iu))) / 2e-6)
        ec = float(np.max(np.abs(gac - np.array(gnc))) / (np.max(np.abs(gnc)) + 1e-12))
        print(f"  {name:>4} D={D:>2}: rel grad err  unitary={eu:.2e}  circuit={ec:.2e}")
        assert eu < 1e-5 and ec < 1e-5, "gradient check FAILED"
    print("[selftest] OK")


# ============================================================================
# driver
# ============================================================================
NONABELIAN = ["S3", "D4", "Q8", "D5", "A4", "D6", "D10", "F21", "S4"]
ABELIAN = ["C6", "C8", "C12"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=NONABELIAN + ABELIAN)
    ap.add_argument("--tasks", nargs="+", default=["generic_W", "offdiag_W"])
    ap.add_argument("--shots", nargs="+", default=SHOT_GRID)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return
    selftest()
    os.makedirs(OUT, exist_ok=True)
    grid = [None if s == "exact" else int(s) for s in a.shots]

    struct, rows = [], []
    for name in a.groups:
        T, m = group_table(name)
        L, Rop = regular_ops(T)
        ab = is_abelian(T)
        bases = [fourier_basis(L, Rop, b) for b in range(N_FIXED_BASES)]
        sr = structure_row(name, T, L, Rop, bases[CANONICAL_BASIS], ab)
        struct.append(sr)
        print(f"[struct] {name:>4} |G|={m} abelian={int(ab)} irreps={sr['irrep_dims']} "
              f"blocks={sr['gft_block_sizes']} offdiag={sr['max_offdiag_after_fourier']:.3e} "
              f"t2err={sr['modeF_matches_t2_err']}")
        for task in a.tasks:
            for S in grid:
                t0 = time.time()
                cell = [run_cell(name, T, L, Rop, bases, task, S, sd, ab) for sd in a.seeds]
                rows.extend(cell)
                d = np.array([c["delta_circ"] for c in cell])
                se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
                sig = float(np.mean([c["label_signal_scale"] for c in cell]))
                v = ("vacuous_excluded" if sig < VACUITY_TOL
                     else ("CONFIRMS_CELL" if d.mean() > 2 * se else "KILLS_CELL"))
                print(f"  {name:>4} {task:>9} S={str(S):>5} | "
                      f"modeF={np.mean([c['acc_test_modeF_fixed'] for c in cell]):.4f} "
                      f"circ={np.mean([c['acc_test_learned_circuit'] for c in cell]):.4f} "
                      f"unit={np.mean([c['acc_test_learned_unitary'] for c in cell]):.4f} "
                      f"mfhead={np.mean([c['acc_test_modeF_plus_head'] for c in cell]):.4f} "
                      f"| d_circ={d.mean():+.4f} 2se={2*se:.4f} -> {v} "
                      f"[{time.time()-t0:.0f}s]")

    sfx = (("_" + a.tag) if a.tag else "")
    cols = list(dict.fromkeys(k for r in rows for k in r))
    p1 = os.path.join(OUT, f"u1_shotmatched_percell{sfx}.csv")
    with open(p1, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    # ---------------- summary + pre-registered per-cell verdicts ----------------
    srows = []
    for name in a.groups:
        for task in a.tasks:
            for S in grid:
                sub = [r for r in rows if r["group"] == name and r["task"] == task
                       and r["shots"] == ("exact" if S is None else S)]
                if not sub:
                    continue
                sig = float(np.mean([r["label_signal_scale"] for r in sub]))
                vac = sig < VACUITY_TOL
                rec = dict(group=name, abelian=sub[0]["abelian"], order=sub[0]["order"],
                           task=task, shots=("exact" if S is None else S), n_seeds=len(sub),
                           label_signal_scale=sig, cell_is_vacuous=int(vac))
                for tag in ("modeF_fixed", "modeF_plus_head", "modeF_bestbasis",
                            "learned_unitary", "learned_circuit", "modeF_totalmatched"):
                    for sp in ("train", "test"):
                        v = np.array([r[f"acc_{sp}_{tag}"] for r in sub], float)
                        rec[f"mean_acc_{sp}_{tag}"] = float(v.mean())
                        rec[f"se_acc_{sp}_{tag}"] = float(v.std(ddof=1) / np.sqrt(len(v)))
                for key, lab in (("delta_circ", "circ"), ("delta_unit", "unit"),
                                 ("delta_circ_totalmatched", "circ_totalmatched")):
                    d = np.array([r[key] for r in sub], float)
                    se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
                    rec[f"mean_delta_{lab}"] = float(d.mean())
                    rec[f"se_delta_{lab}"] = se
                    rec[f"two_se_delta_{lab}"] = 2 * se
                    rec[f"verdict_{lab}"] = ("vacuous_excluded" if vac else
                                             ("CONFIRMS_CELL" if d.mean() > 2 * se
                                              else "KILLS_CELL"))
                rec["mean_readout_fit_shots_learned_circuit"] = (
                    "" if S is None else float(np.mean(
                        [float(r["readout_fit_shots_learned_circuit"]) for r in sub])))
                rec["deploy_shots_per_arm"] = "" if S is None else S * (N_TRAIN + N_TEST)
                rec["shots_per_state_totalmatched"] = (
                    "" if S is None else sub[0]["shots_per_state_totalmatched"])
                srows.append(rec)
    p2 = os.path.join(OUT, f"u1_shotmatched_summary{sfx}.csv")
    with open(p2, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0].keys()))
        w.writeheader(); w.writerows(srows)

    p3 = os.path.join(OUT, f"u1_structure_check{sfx}.csv")
    with open(p3, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(struct[0].keys()))
        w.writeheader(); w.writerows(struct)

    # ---------------- track-level verdict, exactly as pre-registered ----------------
    dec = [r for r in srows if r["task"] == DECIDING_TASK and r["shots"] == DECIDING_SHOTS]
    na = [r for r in dec if r["abelian"] == 0 and not r["cell_is_vacuous"]]
    ab = [r for r in dec if r["abelian"] == 1 and not r["cell_is_vacuous"]]
    n_na = sum(1 for r in na if r["verdict_circ"] == "CONFIRMS_CELL")
    n_ab = sum(1 for r in ab if r["verdict_circ"] == "CONFIRMS_CELL")
    if n_na >= NONAB_MAJORITY and n_ab == 0:
        verdict = "CONFIRMS"
    elif n_na >= NONAB_MAJORITY:
        verdict = "MIXED_HARNESS_RED_FLAG"
    else:
        verdict = "KILLS"
    off = {r["group"]: r["max_offdiag_after_fourier"] for r in struct}
    xs = np.array([off[r["group"]] for r in na], float)
    ys = np.array([r["mean_delta_circ"] for r in na], float)
    pear = (float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 and xs.std() > 0 and ys.std() > 0
            else "")
    vrow = dict(deciding_task=DECIDING_TASK, deciding_shots=DECIDING_SHOTS,
                n_nonabelian_tested=len(na), n_nonabelian_confirms=n_na,
                required_nonabelian_confirms=NONAB_MAJORITY,
                n_abelian_tested=len(ab), n_abelian_confirms=n_ab,
                track_verdict=verdict,
                pearson_r_delta_vs_offdiag_weight=pear,
                n_seeds=len(a.seeds), n_train=N_TRAIN, n_test=N_TEST,
                epochs=EPOCHS, restarts=N_RESTARTS_R, circuit_layers=L_CIRC)
    p4 = os.path.join(OUT, f"u1_verdict{sfx}.csv")
    with open(p4, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(vrow.keys()))
        w.writeheader(); w.writerow(vrow)

    print("\n" + "=" * 78)
    print(f"PRE-REGISTERED TRACK VERDICT: {verdict}")
    print(f"  non-abelian CONFIRMS_CELL: {n_na}/{len(na)} (need >= {NONAB_MAJORITY})")
    print(f"  abelian     CONFIRMS_CELL: {n_ab}/{len(ab)} (need 0)")
    print(f"  pearson r (mean_delta_circ vs off-Fourier-diagonal weight): {pear}")
    print("=" * 78)
    for p in (p1, p2, p3, p4):
        print("wrote", p)


if __name__ == "__main__":
    main()
