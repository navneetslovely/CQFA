"""
experiments/spt_measurement_cost/readout_coherent.py
=======================================
Coherent 2-copy joint readouts, computed STATEVECTOR-ONLY. This is the CQFA arm:
the frame register (G_copy = S2 = {I, SWAP}) realizes the joint 2-copy measurement
via a controlled-SWAP off one frame qubit (the swap test / Bell measurement).

OOM GUARDRAIL (primary): we NEVER materialize rho(x)rho or
psi(x)psi as a 4^n object. Every 2-copy expectation uses the exact identities

    Tr[(P (x) P)(rho (x) rho)] = <psi|P|psi>^2               (two_copy_expectation)
    Tr[SWAP (rho (x) rho)]     = Tr(rho^2)                   (purity)

and the simulated 2-copy shots are drawn from single-copy Bell-basis marginals of
the two identical copies -- again 2^n vectors only. `assert_no_4n_object` is the
tripwire the unit test uses to prove no code path here allocates >= 2^(2n) entries.

Group-theory bridge: the swap test IS the coherent frame readout for
G_copy = S2. `V = |0><0|_F (x) I + |1><1|_F (x) SWAP` matches
Representation.controlled_oracle() for the order-2 rep; `frame_register_swaptest`
below builds that explicit circuit on a small toy pair to certify the abstract
oracle and the physical frame agree (mirrors cqfa_layer.postselect_symmetric).
"""
from __future__ import annotations

import numpy as np

# -- single-qubit Bell-measurement eigenvalue tables ---------------------------
# For a Pauli sigma in {I,X,Y,Z} on a symmetric 2-copy pair, one round of the
# per-qubit Bell measurement (CNOT_{A->B}, H on A, measure both) returns a 2-bit
# outcome (a,b). The measured value of sigma(x)sigma on that Bell outcome is a
# deterministic +/-1 sign; the product over sites is an unbiased +/-1 sample of
# <P>^2 = Tr[(P(x)P) rho(x)rho].
#
# Bell outcome (a,b) <-> joint eigenvalues (X_A X_B = sx, Z_A Z_B = sz):
#   Standard Bell measurement: sx = (-1)^a  is the X_AX_B eigenvalue after the
#   H on A;  sz = (-1)^b is the Z_AZ_B eigenvalue.  Y_AY_B = -(X_AX_B)(Z_AZ_B)
#   because Y(x)Y = -(X(x)X)(Z(x)Z) as sigma(x)sigma operators.
#
# Below we implement the physically- sampler directly from the two-copy
# marginal (§`bell_sample`), which does not need this table; the table documents
# the sign convention used to reduce each Bell outcome to a +/-1 value of <P>^2.


def two_copy_expectation(psi: np.ndarray, P: np.ndarray) -> float:
    """Exact 2-copy expectation Tr[(P(x)P)(rho(x)rho)] = <psi|P|psi>^2.

    Statevector-only: computes <psi|P|psi> on the 2^n vector and squares it. NEVER
    forms P(x)P or rho(x)rho.
    """
    e = float(np.real(psi.conj() @ (P @ psi)))
    return e * e


def purity(psi: np.ndarray) -> float:
    """Tr(rho^2) for a pure state = 1.0 (statevector-only)."""
    ov = complex(psi.conj() @ psi)
    return float(np.real(ov * np.conj(ov)))


def purity_mixture(states, weights) -> float:
    """Tr(rho^2) for rho = sum_k w_k |psi_k><psi_k| via Tr(rho^2)=sum_{jk} w_j w_k |<psi_j|psi_k>|^2.

    Statevector-only (never forms rho); used by the test to check the swap-test
    identity on a genuine mixture.
    """
    states = list(states)
    w = np.asarray(weights, dtype=float)
    p = 0.0
    for j in range(len(states)):
        for k in range(len(states)):
            ov = complex(states[j].conj() @ states[k])
            p += w[j] * w[k] * (abs(ov) ** 2)
    return float(p)


def purity_swaptest(psi_a: np.ndarray, psi_b: np.ndarray | None = None) -> float:
    """Swap-test value <Z_F> = Tr[SWAP (rho_A (x) rho_B)] = |<psi_A|psi_B>|^2.

    For psi_b=None (identical pure copies) returns Tr(rho^2)=1. Statevector-only.
    """
    if psi_b is None:
        psi_b = psi_a
    ov = complex(psi_a.conj() @ psi_b)
    return float(abs(ov) ** 2)


# -- simulated 2-copy shots (Bell sampler), statevector-only -------------------
def bell_sample(psi: np.ndarray, P: np.ndarray, rng: np.random.Generator,
                n_shots: int) -> np.ndarray:
    """Unbiased +/-1 samples of <P>^2 from a coherent 2-copy Bell readout.

    Physically each sample is one round of per-site Bell measurements on the two
    copies. We simulate it statevector-only using the fact that a single Bell round
    yields, for Pauli P, a +/-1 random variable whose mean is exactly <P>^2 and
    whose per-shot variance is 1 - <P>^4 (n-INDEPENDENT). We sample directly from
    that Bernoulli(p) with p = (1+<P>^2)/2 -- equivalent in distribution to the
    site-wise Bell circuit, without ever forming a 2n-qubit object.

    Returns an (n_shots,) array of +/-1 values.
    """
    e2 = two_copy_expectation(psi, P)          # <P>^2 in [0,1] for Hermitian P w/ P^2=I
    e2 = min(1.0, max(-1.0, e2))
    p_plus = 0.5 * (1.0 + e2)                  # P(sample = +1)
    draws = rng.random(n_shots)
    return np.where(draws < p_plus, 1.0, -1.0)


def bell_estimate(psi: np.ndarray, P: np.ndarray, rng: np.random.Generator,
                  n_shots: int) -> float:
    """Monte-Carlo estimate of <P>^2 from n_shots coherent 2-copy Bell samples.

    n_shots here counts 2-copy SHOTS; each consumes 2 copies (see the budget
    accounting in sample_complexity.py).
    """
    return float(np.mean(bell_sample(psi, P, rng, n_shots)))


def swaptest_sample(psi: np.ndarray, rng: np.random.Generator, n_shots: int) -> np.ndarray:
    """+/-1 samples of the swap test (control qubit Z_F outcomes). Mean = Tr(rho^2).

    Statevector-only: p(+1) = (1 + Tr(rho^2))/2. For a pure copy Tr(rho^2)=1 so all
    samples are +1 (the swap test is deterministic on identical pure states).
    """
    pur = purity(psi)
    pur = min(1.0, max(-1.0, pur))
    p_plus = 0.5 * (1.0 + pur)
    draws = rng.random(n_shots)
    return np.where(draws < p_plus, 1.0, -1.0)


def coherent_feature(psi: np.ndarray, strings, rng: np.random.Generator | None = None,
                     n_shots: int | None = None) -> np.ndarray:
    """Feature vector [<P>^2 for P in strings].

    If (rng, n_shots) are given, each entry is the finite-shot Bell estimate;
    otherwise the exact <P>^2. Statevector-only throughout.
    """
    out = np.empty(len(strings), dtype=float)
    for k, P in enumerate(strings):
        if rng is None or n_shots is None:
            out[k] = two_copy_expectation(psi, P)
        else:
            out[k] = bell_estimate(psi, P, rng, n_shots)
    return out


# -- explicit frame-register swap test (small toy, certifies the oracle) -------
def _swap_matrix(dim_per_copy: int) -> np.ndarray:
    """SWAP on two `dim_per_copy`-dim copies (dense; ONLY used on tiny toy states
    inside the certification test, never in the experiment path)."""
    d = dim_per_copy
    S = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            S[j * d + i, i * d + j] = 1.0
    return S


def frame_register_swaptest(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    """Explicit controlled-SWAP-off-a-frame-qubit circuit -> <Z_F>.

    Builds |+>_F (x) (psi_A (x) psi_B), applies V = |0><0|(x)I + |1><1|(x)SWAP,
    a Hadamard on F, and reads <Z_F>. Returns Tr[SWAP (rho_A(x)rho_B)] = |<A|B>|^2.

    NOTE: this DOES form the 2-copy vector -- it is a *certification* routine only,
    meant to be called on TINY toy states (dim_per_copy small, e.g. 4) to prove the
    abstract oracle and the physical frame agree. The experiment path never calls it.
    """
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
    """Context manager that monkeypatches np.zeros/np.empty/np.kron to raise if any
    allocation would create an array with >= 2^(2n) elements. Used by
    test_no_4n_object to prove the coherent path stays statevector-only.
    """

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
