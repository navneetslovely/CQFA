"""
experiments/discrete_log/learners.py
===============================
Two learners for the LAT discrete-log concept class (experiments/discrete_log):

  1. the STRUCTURED learner -- a covariant quantum kernel over the cyclic log
     group C_{p-1}, the CQFA/LAT-style feature map that is AWARE of the log
     structure; and
  2. the CLASSICAL-on-raw-x control -- strong classical learners that see only
     the raw integer x (never log_g(x)), so they cannot represent the concept.

Both are trained with the SAME kernel-SVM / classifier machinery so the only
difference is the *feature map* (log-structure-aware vs raw-x).  This isolates
the inductive-bias effect the LAT theorem is about.

STRUCTURED (covariant C_{p-1}) kernel -- the CQFA/LAT construction
------------------------------------------------------------------------
This is the *covariant quantum kernel* of Glick et al. (Nat. Phys. 20, 479, 2024)
specialised to the cyclic group C_{p-1} that LAT (Nat. Phys. 17, 1013, 2021) use.
Map each datum x to a state produced by the cyclic-group unitary raised to the
data's log coordinate acting on a fixed fiducial state |phi_0>:

    |psi_x> = R^{log_g(x)} |phi_0>,      R = clock (shift) generator of C_{p-1},

and take the fidelity kernel  K(x,x') = |<psi_x|psi_x'>|^2.  Because R is a
cyclic shift, K depends ONLY on  (log_g(x) - log_g(x')) mod (p-1)  -- the group
element separating the two data in log space.  This is exactly the covariance
that lets the kernel resolve the half-arc concept.  The map uses the verified
`cqfa.group.cyclic(p-1)` representation, so the "quantum" feature map is the
same C_{p-1} rep the rest of the framework uses.

We expose the kernel via its EXACT log difference (brute-forced at small p) so
the numbers are traceable; the state construction above is what a quantum device
would run and reproduces the same Gram matrix (asserted in the unit tests).

CLASSICAL-on-raw-x -- the control
---------------------------------
The classical learners get feature vectors built from x itself (normalised x and
its residues), never from log_g(x).  Recovering the log is the DLP; at small p we
CAN brute-force it, but the control is *forbidden* to -- that is the whole point:
it stands in for a poly-time classical learner that cannot solve DLP.  We use a
strong RBF-SVM plus k-NN so the control is a fair, tuned classical baseline, not
a straw man.

INTEGRITY: no scaling claim.  This is a small-p proof-of-principle that the
inductive bias (log/group structure) is what separates the learners; the
asymptotic classical hardness is the CITED LAT theorem under DLP.
"""
from __future__ import annotations

import numpy as np

from cqfa.group import cyclic
from .dlog_concept import DlogConcept


# ---------------------------------------------------------------------------
# the covariant C_{p-1} quantum feature map (STRUCTURED learner)
# ---------------------------------------------------------------------------
def fiducial_state(order: int, bandwidth: float = 1.5) -> np.ndarray:
    """A fixed, deterministic *band-limited* fiducial state |phi_0> in C^order.

    Design (the primary choice).  The covariant fidelity kernel is
    ``K(x,x') = |<phi_0| R^{d} |phi_0>|^2`` with ``d = log(x)-log(x') mod order``.
    Writing ``|phi_0>`` in the shift (Fourier) eigenbasis with mode weights
    ``|a_k|^2``, one gets ``<phi_0|R^d|phi_0> = sum_k |a_k|^2 omega^{k d}`` -- the
    kernel is the (inverse) Fourier transform of the mode-power spectrum.  A
    *random* fiducial spreads power over all modes, making ``K(d)`` a near-delta
    (K=1 at d=0, ~0 else): that kernel only memorises seen log values and cannot
    interpolate the half-arc from few samples.  Concentrating power on the LOW
    circular frequencies (a Gaussian taper of width `bandwidth`) makes ``K(d)`` a
    smooth, wide, monotone-decaying bump whose width matches the arc scale, so
    nearby-in-log points get similar predictions -- the correct covariant
    inductive bias.  `bandwidth` is thus the single kernel hyperparameter; it is
    fixed deterministically here (NOT tuned on test labels).

    Returned state is unit-norm; real (even spectrum) up to normalisation.
    """
    k = np.arange(order)
    circ_freq = np.minimum(k, order - k)          # circular frequency |k|
    weight = np.exp(-(circ_freq ** 2) / (2.0 * bandwidth ** 2))
    amp = np.sqrt(weight / weight.sum()).astype(complex)   # |a_k| on Fourier modes
    # computational-basis state = inverse DFT of the mode amplitudes
    phi = np.fft.ifft(amp) * np.sqrt(order)
    return phi / np.linalg.norm(phi)


def covariant_states(concept: DlogConcept, xs: np.ndarray,
                     phi0: np.ndarray) -> np.ndarray:
    """|psi_x> = R^{log_g(x)} |phi_0> for each x, R = cyclic(p-1) shift.

    Returns an (n, p-1) complex array of unit states.  R is the single
    generator (index 1) of the verified `cqfa.group.cyclic(order)` rep, so R^k is
    the k-fold shift; we apply it by rolling the amplitudes (equivalent, exact).
    """
    order = concept.order
    states = np.empty((len(xs), order), dtype=complex)
    for i, x in enumerate(xs):
        k = concept.logs[int(x)]
        # R|j> = |j+1>, so R^k rolls amplitudes forward by k (see test asserting
        # this equals the matrix power of cyclic(order).matrices[1]).
        states[i] = np.roll(phi0, k)
    return states


def covariant_kernel(states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
    """Fidelity Gram matrix K[i,j] = |<psi_a_i | psi_b_j>|^2."""
    overlap = states_a.conj() @ states_b.T
    return np.abs(overlap) ** 2


# ---------------------------------------------------------------------------
# classical raw-x feature maps (CONTROL learner)
# ---------------------------------------------------------------------------
def raw_x_features(concept: DlogConcept, xs: np.ndarray,
                   kind: str = "scalar") -> np.ndarray:
    """Feature vectors from the RAW integer x only (never log_g(x)).

    Two log-blind encodings a poly-time classical learner could use:

    * ``kind="scalar"``  -- smooth polynomial features of x normalised to [-1,1]
      (a metric encoding: assumes nearby integers x behave similarly, which is
      false for this concept, so it cannot resolve the log-arc);
    * ``kind="onehot"``  -- x as a pure categorical one-hot (the cleanest control:
      x is an OPAQUE group element with no exploitable metric; the learner can
      only memorise seen x's and is exactly at chance on unseen x, as the LAT
      theorem predicts for any learner that cannot compute the discrete log).

    Crucially NEITHER encoding reveals ``log_g(x)``: recovering it is the DLP.
    (We deliberately do NOT feed modular residues ``x mod m`` -- at very small p
    those leak via CRT and would flatter the control.)
    """
    xs = np.asarray(xs, dtype=int)
    if kind == "scalar":
        xn = 2.0 * xs / concept.p - 1.0
        return np.stack([xn, xn ** 2, xn ** 3], axis=1)
    if kind == "onehot":
        M = np.zeros((len(xs), concept.p), dtype=float)
        M[np.arange(len(xs)), xs] = 1.0
        return M
    raise ValueError(f"unknown raw-x feature kind '{kind}'")


# ---------------------------------------------------------------------------
# training / evaluation
# ---------------------------------------------------------------------------
def _svm_from_kernel(K_train: np.ndarray, y_train: np.ndarray, C: float = 1.0):
    """Fit an SVC on a precomputed kernel; return the fitted model."""
    from sklearn.svm import SVC
    clf = SVC(kernel="precomputed", C=C)
    clf.fit(K_train, y_train)
    return clf


def structured_accuracy(concept: DlogConcept, x_train, y_train, x_test, y_test,
                        phi0: np.ndarray | None = None, C: float = 1.0) -> float:
    """Test accuracy of the covariant-C_{p-1} kernel SVM (log-structure-aware)."""
    if phi0 is None:
        phi0 = fiducial_state(concept.order)
    s_tr = covariant_states(concept, x_train, phi0)
    s_te = covariant_states(concept, x_test, phi0)
    K_tr = covariant_kernel(s_tr, s_tr)
    K_te = covariant_kernel(s_te, s_tr)
    clf = _svm_from_kernel(K_tr, y_train, C=C)
    return float(np.mean(clf.predict(K_te) == y_test))


def classical_raw_accuracy(concept: DlogConcept, x_train, y_train, x_test, y_test,
                           method: str = "best") -> float:
    """Best test accuracy over a tuned family of CLASSICAL raw-x learners.

    We report the *strongest* classical baseline -- best over an RBF-SVM sweep and
    k-NN, on BOTH log-blind encodings (`scalar` and `onehot`) -- so the control is
    a fair upper bound on classical-on-raw-x performance, not a straw man.  It
    still cannot beat chance on unseen x because neither encoding exposes the
    discrete log (LAT hardness, cited).
    """
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    candidates = []
    for kind in ("scalar", "onehot"):
        Xtr = raw_x_features(concept, x_train, kind=kind)
        Xte = raw_x_features(concept, x_test, kind=kind)
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
        for C in (0.1, 1.0, 10.0, 100.0):
            for gamma in ("scale", 0.1, 1.0, 10.0):
                clf = SVC(kernel="rbf", C=C, gamma=gamma).fit(Xtr_s, y_train)
                candidates.append(float(np.mean(clf.predict(Xte_s) == y_test)))
        for k in (1, 3, 5):
            if k <= len(y_train):
                clf = KNeighborsClassifier(n_neighbors=k).fit(Xtr_s, y_train)
                candidates.append(float(np.mean(clf.predict(Xte_s) == y_test)))
    if method == "best":
        return float(max(candidates))
    return float(np.mean(candidates))


def chance_accuracy(y_test: np.ndarray) -> float:
    """Majority-class (chance) accuracy baseline for context."""
    if len(y_test) == 0:
        return 0.0
    frac_pos = float(np.mean(y_test == 1))
    return max(frac_pos, 1.0 - frac_pos)
