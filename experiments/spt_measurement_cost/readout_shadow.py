
from __future__ import annotations

import numpy as np


_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)              # X-basis
_HSdg = _H @ np.array([[1, 0], [0, -1j]], dtype=complex)                  # Y-basis: S^dag then H

_ROT = {
    0: _H,                       # measure X  <=> apply H then measure Z
    1: _HSdg,                    # measure Y  <=> apply H S^dag then measure Z
    2: np.eye(2, dtype=complex)  # measure Z
}


_PAULI_LABEL = {0: "X", 1: "Y", 2: "Z"}


def _apply_local_rotations(psi: np.ndarray, n: int, bases: np.ndarray) -> np.ndarray:
    """Apply the per-site basis rotation R_b to |psi> (dense per-qubit apply)."""
    out = psi.reshape([2] * n)
    for q in range(n):
        R = _ROT[int(bases[q])]
        out = np.tensordot(R, out, axes=([1], [q]))
        out = np.moveaxis(out, 0, q)
    return out.reshape(-1)


def _sample_bitstring(psi_rot: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample one computational-basis bitstring from |psi_rot|^2 (multinomial)."""
    probs = np.abs(psi_rot) ** 2
    probs = probs / probs.sum()
    idx = int(rng.choice(len(probs), p=probs))
    bits = np.array([(idx >> (n - 1 - q)) & 1 for q in range(n)], dtype=np.int8)
    return bits


def shadow_snapshots(psi: np.ndarray, rng: np.random.Generator, n_shots: int):
    """Collect n_shots single-copy random-Pauli classical-shadow snapshots.

    Returns (bases, outcomes): each (n_shots, n) int array. bases[k,q] in {0,1,2}
    (X/Y/Z), outcomes[k,q] in {0,1}. Statevector-only sampling.
    """
    n = int(np.log2(len(psi)))
    bases = rng.integers(0, 3, size=(n_shots, n)).astype(np.int8)
    outcomes = np.empty((n_shots, n), dtype=np.int8)
    for k in range(n_shots):
        psi_rot = _apply_local_rotations(psi, n, bases[k])
        outcomes[k] = _sample_bitstring(psi_rot, n, rng)
    return bases, outcomes


def _pauli_support(P_label: dict):
    """Given a Pauli as {site: 'X'|'Y'|'Z'}, return arrays (sites, basis_codes)."""
    sites = sorted(P_label.keys())
    codes = np.array([{"X": 0, "Y": 1, "Z": 2}[P_label[s]] for s in sites], dtype=np.int8)
    return np.array(sites, dtype=int), codes


def _single_snapshot_estimate(bases_row, outcomes_row, sites, codes) -> float:
    """Standard random-Pauli shadow estimator of <P> from ONE snapshot:
        prod_{q in supp(P)} [ 3*(-1)^outcome  if base matches P's Pauli else 0 ].
    Returns 0 whenever any support qubit was not measured in P's Pauli basis.
    """
    val = 1.0
    for s, c in zip(sites, codes):
        if bases_row[s] != c:
            return 0.0
        val *= 3.0 * (1.0 if outcomes_row[s] == 0 else -1.0)
    return val


def estimate_expectation(snaps, P_label: dict) -> float:
    """Unbiased <P> from a batch of shadow snapshots (linear estimator)."""
    bases, outcomes = snaps
    sites, codes = _pauli_support(P_label)
    if len(sites) == 0:
        return 1.0
    vals = np.array([_single_snapshot_estimate(bases[k], outcomes[k], sites, codes)
                     for k in range(len(bases))])
    return float(np.mean(vals))


def _per_snapshot_values(snaps, P_label: dict) -> np.ndarray:
    bases, outcomes = snaps
    sites, codes = _pauli_support(P_label)
    return np.array([_single_snapshot_estimate(bases[k], outcomes[k], sites, codes)
                     for k in range(len(bases))])


def estimate_square_ustat(snaps, P_label: dict) -> float:
  
    v = _per_snapshot_values(snaps, P_label)
    N = len(v)
    if N < 2:
        return float("nan")
    s = v.sum()
    sq = (v * v).sum()
    return float((s * s - sq) / (N * (N - 1)))


# -- Pauli-label helpers for the string operators ------------------------------
def string_pauli_label(n: int, a: int | None = None, b: int | None = None) -> dict:

    if a is None:
        a = 1
    if b is None:
        b = n - 2
  
    from collections import Counter
    zc = Counter()
    xc = Counter()
    for i in range(a, b + 1):
        zc[i - 1] += 1
        xc[i] += 1
        zc[i + 1] += 1
    label = {}
    sites = set(list(zc.keys()) + list(xc.keys()))
    for s in sites:
        z = zc[s] % 2
        x = xc[s] % 2
        if z and x:
            label[s] = "Y"
        elif z:
            label[s] = "Z"
        elif x:
            label[s] = "X"
        # z==0 and x==0 -> identity, omit
    return label


# -- baselines -----------------------------------------------------------------
def direct_pauli_samples(psi: np.ndarray, P_label: dict, rng: np.random.Generator,
                         n_shots: int) -> np.ndarray:
  
    n = int(np.log2(len(psi)))
    sites, codes = _pauli_support(P_label)
    bases = np.full(n, 2, dtype=np.int8)      # non-support qubits: harmless Z basis
    for s, c in zip(sites, codes):
        bases[s] = c
    out = np.empty(n_shots, dtype=float)
    for k in range(n_shots):
        psi_rot = _apply_local_rotations(psi, n, bases)
        bits = _sample_bitstring(psi_rot, n, rng)
        parity = int(np.sum(bits[sites])) % 2
        out[k] = 1.0 if parity == 0 else -1.0
    return out


def known_string_estimate(psi: np.ndarray, n: int, rng: np.random.Generator,
                          n_shots: int) -> float:
   
    return float(np.mean(direct_pauli_samples(psi, string_pauli_label(n), rng, n_shots)))


def local_observable_features(snaps, n: int) -> np.ndarray:
    
    feats = []
    for q in range(n):
        feats.append(estimate_expectation(snaps, {q: "Z"}))
        feats.append(estimate_expectation(snaps, {q: "X"}))
    for q in range(n - 1):
        feats.append(estimate_expectation(snaps, {q: "Z", q + 1: "Z"}))
    return np.array(feats)


def functional_estimate_shadow(snaps, strings_labels) -> float:
   
    total = 0.0
    for lab in strings_labels:
        total += estimate_square_ustat(snaps, lab)
    return float(total)


if __name__ == "__main__":
    from experiments.spt_measurement_cost import spt_model as M
    n = 6
    psi = M.cluster_state(n)
    rng = np.random.default_rng(0)
    snaps = shadow_snapshots(psi, rng, 40000)
    lab = string_pauli_label(n)
    print("string label      :", lab)
    exact = M.string_order(psi, n)
    print("exact <O_string>  :", exact)
    print("shadow <O_string> :", estimate_expectation(snaps, lab))
    print("shadow <O>^2 ustat:", estimate_square_ustat(snaps, lab), " exact:", exact ** 2)
    print("direct <O> (known):", known_string_estimate(psi, n, np.random.default_rng(1), 40000),
          " exact:", exact)
