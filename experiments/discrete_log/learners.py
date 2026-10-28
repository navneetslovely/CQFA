
from __future__ import annotations

import numpy as np

from cqfa.group import cyclic
from .dlog_concept import DlogConcept


# ---------------------------------------------------------------------------
# the covariant C_{p-1} quantum feature map (STRUCTURED learner)
# ---------------------------------------------------------------------------
def fiducial_state(order: int, bandwidth: float = 1.5) -> np.ndarray:
  
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
