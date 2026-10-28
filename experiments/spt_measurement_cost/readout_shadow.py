"""
experiments/spt_measurement_cost/readout_shadow.py
====================================
Single-copy classical-shadow baseline (Huang-Kueng-Preskill 2020, random-Pauli
flavour). This is the arm the coherent 2-copy readout beats on the nonlinear /
unknown-string task.

Everything is statevector-only: to draw a single-copy measurement outcome we rotate
|psi> by the chosen local Cliffords (one of {X,Y,Z}-basis rotations per site) and
sample a bitstring from |amplitudes|^2 (multinomial). No density matrix is formed.

Estimators:
  * estimate_expectation(snaps, P)  -> unbiased <P> (linear; per-shot variance O(1)
    for a SINGLE known Pauli -> the "known-string control" that stays FLAT in n).
  * estimate_square_ustat(snaps, P) -> unbiased <P>^2 via a U-statistic over pairs
    of independent snapshots. For the exponentially-large string family / purity the
    single-copy sample complexity is 2^Omega(n) (HKP21, CCHL) -> the arm that BLOWS UP.
  * local_observable_features(...)  -> low-weight (single-site + nearest-neighbour)
    shadow estimates only; provably cannot separate the phases (control 1, FAILS).
  * functional_estimate_shadow(...) -> single-copy estimate of the Option-A nonlinear
    functional F(rho) = sum_{P in S_string} <P>^2 (blows up).
"""
from __future__ import annotations

import numpy as np

# single-qubit basis rotations (map measurement basis to Z basis) --------------
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)              # X-basis
_HSdg = _H @ np.array([[1, 0], [0, -1j]], dtype=complex)                  # Y-basis: S^dag then H
# Rotation that takes the |psi> so that a Z-basis measurement realizes the chosen
# Pauli-basis measurement. For basis b in {0:X,1:Y,2:Z} we apply R_b to |psi>.
_ROT = {
    0: _H,                       # measure X  <=> apply H then measure Z
    1: _HSdg,                    # measure Y  <=> apply H S^dag then measure Z
    2: np.eye(2, dtype=complex)  # measure Z
}
# Post-measurement single-shot inverse-channel snapshot ingredient:
# for outcome bit s in {0,1} and basis b, the snapshot single-qubit operator is
#   3 * |b,s><b,s| - I
# where |b,s> is the eigenstate of Pauli_b with eigenvalue (-1)^s.
# The estimator of <sigma_b'> from that snapshot is:
#   tr[(3|b,s><b,s| - I) sigma_b'] = 3*<b,s|sigma_b'|b,s> - tr(sigma_b')
# tr(sigma)=0 for X,Y,Z, and <b,s|sigma_b'|b,s> is (-1)^s if b'==b else 0.
# => single-qubit estimator contribution is  3*(-1)^s  if b'==b else 0 (for a
#    matching Pauli), or 1 for identity. This is the standard random-Pauli shadow.

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
    """Unbiased <P>^2 via the U-statistic over distinct snapshot pairs:
        (1/(N(N-1))) sum_{j != k} v_j v_k    where v = per-snapshot <P> estimate.
    E[v_j v_k] = <P>^2 for j != k (independent snapshots). This is the single-copy
    estimator of the NONLINEAR quantity; its variance carries the 2^Omega(n) cost.
    """
    v = _per_snapshot_values(snaps, P_label)
    N = len(v)
    if N < 2:
        return float("nan")
    s = v.sum()
    sq = (v * v).sum()
    return float((s * s - sq) / (N * (N - 1)))


# -- Pauli-label helpers for the string operators ------------------------------
def string_pauli_label(n: int, a: int | None = None, b: int | None = None) -> dict:
    """The Y-decorated string O_string(a,b) as {site: Pauli} for shadow estimation.

    For the bulk interval a=1,b=n-2 this is Z_0 Y_1 X_2..X_{n-3} Y_{n-2} Z_{n-1}.
    Derived from the telescoping of K_i = Z_{i-1} X_i Z_{i+1}; verified by
    test_readout::test_string_label_matches_operator.
    """
    if a is None:
        a = 1
    if b is None:
        b = n - 2
    # Telescope: track Pauli parity per site over the product of K_i.
    parity = {}  # site -> set of {'X','Z'} contributions counted mod nothing; we combine
    # Build by multiplying single-site Paulis with Z*X = -iY bookkeeping (sign dropped;
    # the operator sign is handled by string_operator; here we only need the support).
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
    """+/-1 samples of a SINGLE KNOWN Pauli string P, measured DIRECTLY in its own
    eigenbasis (every shot measures P). Mean = <P>, per-shot variance 1 - <P>^2 <= 1,
    n-INDEPENDENT. This is control 2: knowing the string, single-copy is easy.

    Statevector-only: rotate each support qubit to the Z basis for its Pauli, sample
    a bitstring, and take the parity (-1)^(sum of support bits) as the +/-1 outcome.
    """
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
    """CONTROL 2: single-copy estimate of the ONE KNOWN string <O_string> by DIRECT
    Pauli measurement. Easy, n-independent per-shot variance -> stays FLAT."""
    return float(np.mean(direct_pauli_samples(psi, string_pauli_label(n), rng, n_shots)))


def local_observable_features(snaps, n: int) -> np.ndarray:
    """CONTROL 1: low-weight (single-site Z + nearest-neighbour ZZ) shadow features.

    These cannot separate the phases at any budget. Returns a feature
    vector for a downstream local-observable classifier.
    """
    feats = []
    for q in range(n):
        feats.append(estimate_expectation(snaps, {q: "Z"}))
        feats.append(estimate_expectation(snaps, {q: "X"}))
    for q in range(n - 1):
        feats.append(estimate_expectation(snaps, {q: "Z", q + 1: "Z"}))
    return np.array(feats)


def functional_estimate_shadow(snaps, strings_labels) -> float:
    """THE baseline: single-copy estimate of the nonlinear functional
        F(rho) = sum_{P in S_string} <P>^2
    via the U-statistic per string. Blows up (2^Omega(n)) as |S_string| grows.
    """
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
