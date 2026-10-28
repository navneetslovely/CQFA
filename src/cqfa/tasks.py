"""Learning-task construction.

``coherent_teacher_task`` labels each state by thresholding a fixed coherent
teacher feature

    c(psi) = <Pi psi| M |Pi psi> / || Pi psi ||^2,

which depends on cross-orbit interference. Classical frame averaging observes
only the orbit outputs {<psi| U_g^dag M U_g |psi>}, so it is
information-theoretically bottlenecked on this concept while the coherent
readout can represent it.

The remaining constructors cover orbit interference, symmetric versus
symmetry-broken targets, and states derived from symmetry-protected order.
"""
from __future__ import annotations

import numpy as np

from .group import Representation
from . import frame as L


def random_states(dim: int, n: int, rng: np.random.Generator) -> np.ndarray:
    V = rng.standard_normal((n, dim)) + 1j * rng.standard_normal((n, dim))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V


def random_hermitian(dim: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    return (A + A.conj().T) / 2.0


def coherent_teacher_task(rep: Representation, n_samples: int,
                          rng: np.random.Generator, teacher_M: np.ndarray):
    """Sample states and label by thresholding the coherent teacher feature.

    Returns (X, y): X an (n_samples, dim) complex array of states, y in {0,1}.
    States annihilated by Pi (post-selection prob ~0) are skipped.
    """
    states, feats = [], []
    while len(states) < n_samples:
        psi = random_states(rep.dim, 1, rng)[0]
        c = L.coherent_feature(psi, rep, teacher_M)
        if c is None:
            continue
        states.append(psi)
        feats.append(c)
    feats = np.array(feats)
    y = (feats > np.median(feats)).astype(int)
    return np.array(states), y


# ===========================================================================
# Benchmark tasks: the same (X, y) contract as coherent_teacher_task.
# Each returns X (n_samples, rep.dim) complex unit states, y in {0,1}^n_samples,
# median-balanced. All randomness flows through the passed rng (deterministic).
# ===========================================================================
def orbit_interference_task(rep: Representation, n_samples: int,
                            rng: np.random.Generator, teacher_M: np.ndarray):
    """T2 -- the sharpest expressivity separation.

    Label = sign (vs median) of the *coherent surplus* itself:
        surplus(psi) = f_A(psi) - f_B(psi)
    where f_A = cqfa_readout (Mode A, coherent amplitude average) and
          f_B = classical_fa_readout (Mode B, orbit-output average).
    The decision boundary IS the off-diagonal interference term that Mode B is,
    by construction, blind to. Classical-FA therefore cannot separate it above
    chance; CQFA can. `advantage_claim = expressivity`.

    Pi-annihilated states (f_A is None) are skipped, as in the coherent-teacher
    task.
    """
    states, surplus = [], []
    while len(states) < n_samples:
        psi = random_states(rep.dim, 1, rng)[0]
        fa, _p = L.cqfa_readout(psi, rep, teacher_M)
        if fa is None:                       # annihilated by Pi
            continue
        fb = L.classical_fa_readout(psi, rep, teacher_M)
        states.append(psi)
        surplus.append(fa - fb)
    surplus = np.array(surplus)
    y = (surplus > np.median(surplus)).astype(int)
    return np.array(states), y


def symmetry_vs_broken_task(rep: Representation, n_samples: int,
                            rng: np.random.Generator,
                            q_lo_quantile: float = 0.33,
                            q_hi_quantile: float = 0.67,
                            calib_pool: int = 400):
    """T3 -- equivariance-as-inductive-bias.

    Two clouds graded by the frame asymmetry order parameter
        q(psi) = 1 - ||Pi psi||^2   (cqfa_layer.frame_asymmetry).
    Class 0 = low-q (symmetric-ish), class 1 = high-q (broken).

    Robustness note: the *absolute* q distribution is rep-dependent (for larger
    orbits e.g. C8 the random-state q concentrates near 1, so fixed absolute
    bands q_lo=0.15/q_hi=0.55 from the spec are unreachable). We therefore use
    per-rep QUANTILE thresholds calibrated on a fresh pool of random states:
    below the q_lo quantile -> class 0, above the q_hi quantile -> class 1, the
    ambiguous middle band rejected. This always populates both classes and stays
    fully deterministic (calibration draws come from the same rng stream).

    Equivariant models (cqfa, classical_fa) should track this order parameter
    cleanly; non-equivariant and augmented act as controls.
    """
    # calibrate quantile cut-points on a pool drawn from the same rng
    pool = random_states(rep.dim, calib_pool, rng)
    qpool = np.array([L.frame_asymmetry(p, rep) for p in pool])
    q_lo = float(np.quantile(qpool, q_lo_quantile))
    q_hi = float(np.quantile(qpool, q_hi_quantile))
    states, labels = [], []
    n_each = n_samples // 2
    n0 = n1 = 0
    max_tries = 200 * n_samples + 10000
    tries = 0
    while (n0 < n_each or n1 < n_samples - n_each) and tries < max_tries:
        tries += 1
        psi = random_states(rep.dim, 1, rng)[0]
        q = L.frame_asymmetry(psi, rep)
        if q <= q_lo and n0 < n_each:
            states.append(psi); labels.append(0); n0 += 1
        elif q >= q_hi and n1 < n_samples - n_each:
            states.append(psi); labels.append(1); n1 += 1
    return np.array(states), np.array(labels, dtype=int)


def _random_local_perturb(psi: np.ndarray, n: int, rng: np.random.Generator,
                          eps: float = 0.25) -> np.ndarray:
    """Apply independent small single-qubit RY/RZ rotations to each of n qubits.

    Keeps the state physical (unit norm) and the task non-trivial but learnable.
    Uses the verified statevector primitives in cqfa.simulator.
    """
    from .simulator import _apply_1q_state, RY, RZ
    out = np.asarray(psi, dtype=complex).copy()
    for q in range(n):
        out = _apply_1q_state(out, RY(eps * rng.standard_normal()), q, n)
        out = _apply_1q_state(out, RZ(eps * rng.standard_normal()), q, n)
    return out / np.linalg.norm(out)


def spt_derived_task(rep: Representation, n_samples: int,
                     rng: np.random.Generator, eps: float = 0.25):
    """A trainable-scale binary task derived from symmetry-protected order.

    rep must be a power-of-two rep on n = log2(dim) qubits (use z2_pauli(n)).
    We sample states interpolating between the exact cluster/SPT ground state
    (h=0, string order = +1) and a trivial paramagnet (h=3, string order ~0),
    each dressed with small random single-qubit rotations. Label = SPT-side (0)
    vs trivial-side (1).

    Uses the ground-state primitives in
    experiments/spt_measurement_cost/spt_physics_reference.py. The two base states differ (overlap 0.29-0.54 for n=2..4),
    so the concept is learnable even where string order alone would not separate
    (e.g. n=2 has no interior stabilizer site).

    Scope: this is an expressivity and inductive-bias demonstration on
    SPT-derived states, recorded as `advantage_claim = inductive_bias`. It is
    distinct from the measurement-complexity result in
    experiments/spt_measurement_cost/, which concerns the readout protocol.
    """
    from experiments.spt_measurement_cost.spt_physics_reference import ground_state
    n = int(round(np.log2(rep.dim)))
    if 2 ** n != rep.dim:
        raise ValueError(f"spt_derived needs a power-of-two rep dim, got {rep.dim}")
    spt = ground_state(n, h=0.0)             # cluster state (string order +1)
    triv = ground_state(n, h=3.0)            # paramagnet (string order ~0)
    states, labels = [], []
    for _ in range(n_samples):
        if rng.random() < 0.5:
            base, lab = spt, 0
        else:
            base, lab = triv, 1
        states.append(_random_local_perturb(base, n, rng, eps=eps))
        labels.append(lab)
    return np.array(states), np.array(labels, dtype=int)


# ---------------------------------------------------------------------------
# task registry + factory (used by the benchmark sweep driver)
# ---------------------------------------------------------------------------
TASK_REGISTRY = {
    "coherent_teacher":   coherent_teacher_task,
    "orbit_interference": orbit_interference_task,
    "symmetry_vs_broken": symmetry_vs_broken_task,
    "spt_derived":        spt_derived_task,
    # genomics_rc is a dataset-backed task supplied by the driver, not here.
}


def make_task(name: str, rep: Representation, teacher_seed: int = 12345):
    """Return a closure (n_samples, rng) -> (X, y) for the sweep driver.

    The teacher observable M (for the two teacher-based tasks) is fixed by
    `teacher_seed` so the *concept* is identical across data seeds -- the correct
    experimental design already used in cqfa.train.run_experiment.
    """
    fn = TASK_REGISTRY[name]
    if name in ("coherent_teacher", "orbit_interference"):
        M = random_hermitian(rep.dim, np.random.default_rng(teacher_seed))
        return lambda n, rng: fn(rep, n, rng, M)
    if name == "spt_derived":
        return lambda n, rng: fn(rep, n, rng)
    if name == "symmetry_vs_broken":
        return lambda n, rng: fn(rep, n, rng)
    raise KeyError(f"unknown registry task '{name}'")
