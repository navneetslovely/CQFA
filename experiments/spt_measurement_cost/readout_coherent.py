
from __future__ import annotations

import numpy as np



def two_copy_expectation(psi: np.ndarray, P: np.ndarray) -> float:
    
    e = float(np.real(psi.conj() @ (P @ psi)))
    return e * e


def purity(psi: np.ndarray) -> float:
    """Tr(rho^2) for a pure state = 1.0 (statevector-only)."""
    ov = complex(psi.conj() @ psi)
    return float(np.real(ov * np.conj(ov)))


def purity_mixture(states, weights) -> float:
    
    states = list(states)
    w = np.asarray(weights, dtype=float)
    p = 0.0
    for j in range(len(states)):
        for k in range(len(states)):
            ov = complex(states[j].conj() @ states[k])
            p += w[j] * w[k] * (abs(ov) ** 2)
    return float(p)


def purity_swaptest(psi_a: np.ndarray, psi_b: np.ndarray | None = None) -> float:
    
    if psi_b is None:
        psi_b = psi_a
    ov = complex(psi_a.conj() @ psi_b)
    return float(abs(ov) ** 2)


# -- simulated 2-copy shots (Bell sampler), statevector-only -------------------
def bell_sample(psi: np.ndarray, P: np.ndarray, rng: np.random.Generator,
                n_shots: int) -> np.ndarray:
    
    e2 = two_copy_expectation(psi, P)          # <P>^2 in [0,1] for Hermitian P w/ P^2=I
    e2 = min(1.0, max(-1.0, e2))
    p_plus = 0.5 * (1.0 + e2)                  # P(sample = +1)
    draws = rng.random(n_shots)
    return np.where(draws < p_plus, 1.0, -1.0)


def bell_estimate(psi: np.ndarray, P: np.ndarray, rng: np.random.Generator,
                  n_shots: int) -> float:
    
    return float(np.mean(bell_sample(psi, P, rng, n_shots)))


def swaptest_sample(psi: np.ndarray, rng: np.random.Generator, n_shots: int) -> np.ndarray:
    
    pur = purity(psi)
    pur = min(1.0, max(-1.0, pur))
    p_plus = 0.5 * (1.0 + pur)
    draws = rng.random(n_shots)
    return np.where(draws < p_plus, 1.0, -1.0)


def coherent_feature(psi: np.ndarray, strings, rng: np.random.Generator | None = None,
                     n_shots: int | None = None) -> np.ndarray:
    
    out = np.empty(len(strings), dtype=float)
    for k, P in enumerate(strings):
        if rng is None or n_shots is None:
            out[k] = two_copy_expectation(psi, P)
        else:
            out[k] = bell_estimate(psi, P, rng, n_shots)
    return out


# -- explicit frame-register swap test (small toy, certifies the oracle) -------
def _swap_matrix(dim_per_copy: int) -> np.ndarray:
  
    d = dim_per_copy
    S = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            S[j * d + i, i * d + j] = 1.0
    return S


def frame_register_swaptest(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    
    d = len(psi_a)
    assert d == len(psi_b)
    two_copy = np.kron(psi_a, psi_b)                     # (d*d,)
    S = _swap_matrix(d)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    Iq = np.eye(d * d, dtype=complex)
    # frame in |+>
    frame_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    state = np.kron(frame_plus, two_copy)               # (2*d*d,)
    # controlled-SWAP V (frame is the top qubit)
    V = np.zeros((2 * d * d, 2 * d * d), dtype=complex)
    V[: d * d, : d * d] = Iq                             # |0><0| (x) I
    V[d * d:, d * d:] = S                                # |1><1| (x) SWAP
    state = V @ state
    # Hadamard on frame
    Hfull = np.kron(H, Iq)
    state = Hfull @ state
    # <Z_F> = P(frame=0) - P(frame=1)
    p0 = float(np.real(state[: d * d].conj() @ state[: d * d]))
    p1 = float(np.real(state[d * d:].conj() @ state[d * d:]))
    return p0 - p1


# -- OOM tripwire --------------------------------------------------------------
class FourNTripwire:
    

    def __init__(self, n: int):
        self.limit = 2 ** (2 * n)
        self._orig = {}

    def _wrap(self, fn, name):
        def guarded(*args, **kwargs):
            arr = fn(*args, **kwargs)
            if getattr(arr, "size", 0) >= self.limit:
                raise MemoryError(
                    f"{name} allocated array of size {arr.size} >= 2^(2n)={self.limit} "
                    "(forbidden 4^n object in the 2-copy path)")
            return arr
        return guarded

    def __enter__(self):
        for name in ("zeros", "empty", "kron", "ones"):
            self._orig[name] = getattr(np, name)
            setattr(np, name, self._wrap(self._orig[name], f"np.{name}"))
        return self

    def __exit__(self, *exc):
        for name, fn in self._orig.items():
            setattr(np, name, fn)
        return False


if __name__ == "__main__":
    from experiments.spt_measurement_cost import spt_model as M
    n = 8
    psi = M.cluster_state(n)
    O = M.string_operator(n)
    print("exact <O_string>^2 :", two_copy_expectation(psi, O))
    print("purity (pure)      :", purity(psi))
    print("swaptest identical :", purity_swaptest(psi))
    rng = np.random.default_rng(0)
    print("bell estimate      :", bell_estimate(psi, O, rng, 20000))
    # frame register certification on a tiny toy pair
    toy = np.array([0.6, 0.0, 0.0, 0.8], dtype=complex)
    print("frame swaptest toy :", frame_register_swaptest(toy, toy),
          " vs |<a|b>|^2 =", purity_swaptest(toy))
