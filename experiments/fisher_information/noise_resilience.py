"""
experiments/fisher_information/noise_resilience.py
================================================
Resilience of the Mode F (frame Fourier) readout under realistic noise.
Density-matrix exact, no shots: depolarizing noise on every frame qubit before the
Fourier readout, plus classical readout bit-flips on the measured irrep label.

Question answered (, whatever the curve says): how fast does the Mode-F
hidden-shift accuracy degrade as noise grows, and does the A/B-blind vs F-sees
separation survive device-scale noise rates (~0.1-1% depol, ~1-3% readout)?

Output: results/fisher_information/noise_resilience.csv .
"""
from __future__ import annotations

import argparse
import csv
import os
from itertools import combinations

import numpy as np

from .fourier_mode import shift, _rand_state, _nearest_centroid_acc, hidden_shift_dataset


# ---------------------------------------------------------------------------
# noisy Mode F: joint frame(x)data density matrix -> depol on frame qubits ->
# Fourier-basis frame measurement -> irrep probabilities + data components
# ---------------------------------------------------------------------------
def _joint_state(psi: np.ndarray, N: int) -> np.ndarray:
    """|Phi> = (1/sqrt N) sum_g |g> (x) U_g |psi>  (dim N*N, frame major)."""
    d = len(psi)
    phi = np.zeros(N * d, dtype=complex)
    for g in range(N):
        phi[g * d:(g + 1) * d] = shift(N, g) @ psi
    return phi / np.sqrt(N)


def _depol_frame(rho: np.ndarray, N: int, d: int, p: float) -> np.ndarray:
    """Single-qubit depolarizing (strength p) on EACH of the log2(N) frame qubits.
    Kraus on the frame subsystem; data untouched."""
    if p <= 0:
        return rho
    nq = int(np.log2(N))
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    for q in range(nq):
        ops = []
        for (P, w) in ((I2, 1 - p), (X, p / 3), (Y, p / 3), (Z, p / 3)):
            mats = [I2] * nq
            mats[q] = P
            F = np.array([[1.0 + 0j]])
            for m in mats:
                F = np.kron(F, m)
            ops.append((np.kron(F, np.eye(d)), w))
        rho = sum(w * (K @ rho @ K.conj().T) for K, w in ops)
    return rho


def _fourier_frame_povm(N: int):
    """Fourier-basis frame vectors |chi_k> = (1/sqrt N) sum_g w^{kg} |g>."""
    w = np.exp(2j * np.pi / N)
    return [np.array([w ** (k * g) for g in range(N)]) / np.sqrt(N) for k in range(N)]


def modeF_noisy(psi: np.ndarray, ref: np.ndarray, N: int, p_dep: float, q_flip: float):
    """Noisy Mode-F features via the OPERATIONAL Hadamard-test construction: an
    ancilla holds (|0>|Phi(ref)> + |1>|Phi(psi)>)/sqrt2; measuring X/Y(anc) x Pi_k
    reads the complex overlap c_k = <v_k^ref | v_k^psi> (exactly the noiseless
    Mode-F phase feature at p=0), and Pi_k on the |1> branch reads p_k. Depolarizing
    acts on every frame qubit; readout bit-flips scramble the measured irrep label
    k for BOTH the probabilities and the complex overlaps."""
    d = len(psi)
    phi_r = _joint_state(ref, N)
    phi_p = _joint_state(psi, N)
    D = N * d
    Psi = np.zeros(2 * D, dtype=complex)                   # anc (x) (frame (x) data)
    Psi[:D] = phi_r / np.sqrt(2)                           # anc=0 branch: reference
    Psi[D:] = phi_p / np.sqrt(2)                           # anc=1 branch: psi
    rho = np.outer(Psi, Psi.conj())
    # depolarize frame qubits identically in both ancilla branches
    rho = np.kron(np.eye(2), np.eye(D)) @ rho              # no-op; clarity
    # apply frame depol on the (frame x data) factor of each anc block
    r00 = _depol_frame(rho[:D, :D], N, d, p_dep)
    r01 = _depol_frame(rho[:D, D:], N, d, p_dep)
    r10 = _depol_frame(rho[D:, :D], N, d, p_dep)
    r11 = _depol_frame(rho[D:, D:], N, d, p_dep)
    chis = _fourier_frame_povm(N)
    probs = np.zeros(N)
    overlaps = np.zeros(N, dtype=complex)
    for k, chi in enumerate(chis):
        Pk = np.kron(chi.reshape(-1, 1), np.eye(d))        # <chi_k| on frame
        # p_k from the psi branch (anc=1): 2 * Tr(<chi_k| r11 |chi_k>)
        probs[k] = float(np.real(np.trace(Pk.conj().T @ r11 @ Pk))) * 2.0
        # complex overlap from the anc coherence block: 2 * Tr(<chi_k| r10 |chi_k>)
        overlaps[k] = 2.0 * np.trace(Pk.conj().T @ r10 @ Pk)
    # classical readout bit-flips on the irrep label k (each of log2 N bits w.p. q)
    if q_flip > 0:
        nq = int(np.log2(N))
        T = np.zeros((N, N))
        for k in range(N):
            for k2 in range(N):
                nb = bin(k ^ k2).count("1")
                T[k2, k] = (q_flip ** nb) * ((1 - q_flip) ** (nq - nb))
        probs = T @ probs
        overlaps = T @ overlaps
    return probs, overlaps


def sample_features(probs, overlaps, shots, rng):
    """Finite-shot estimates — the resilience mechanism. With exact
    expectations a fixed channel is an invertible linear map a classifier undoes;
    the REAL noise cost is the shrunken signal against shot noise. p_k from a
    multinomial (`shots` draws); each Re/Im overlap from `shots` Hadamard-test
    +/-1 outcomes with the exact mean."""
    N = len(probs)
    p = np.clip(np.real(probs), 0, None); p = p / p.sum()
    p_hat = rng.multinomial(shots, p) / shots
    re_hat = np.empty(N); im_hat = np.empty(N)
    for k in range(N):
        for arr, mean in ((re_hat, np.real(overlaps[k])), (im_hat, np.imag(overlaps[k]))):
            mu = float(np.clip(mean, -1.0, 1.0))
            plus = rng.binomial(shots, (1.0 + mu) / 2.0)
            arr[k] = 2.0 * plus / shots - 1.0
    return np.concatenate([p_hat, re_hat, im_hat])


def run(N, m, seeds, p_deps, q_flips, out_dir, shots=500):
    rows = []
    for p_dep in p_deps:
        for q_flip in q_flips:
            accs = []
            for sd in seeds:
                rng = np.random.default_rng(sd)
                phi0, data = hidden_shift_dataset(N, m, rng)
                X = []
                for psi, s, y in data:
                    probs, overlaps = modeF_noisy(psi, phi0, N, p_dep, q_flip)
                    X.append(sample_features(probs, overlaps, shots, rng))
                y = [y for _, _, y in data]
                accs.append(_nearest_centroid_acc(X, y))
            rows.append(dict(N=N, p_depol=p_dep, q_readout=q_flip, shots=shots,
                             acc_modeF_noisy=float(np.mean(accs)),
                             acc_std=float(np.std(accs)), n_seeds=len(seeds)))
            print(f"N={N} depol={p_dep:.3f} readout={q_flip:.3f} shots={shots}: "
                  f"Mode F acc={np.mean(accs):.3f}+/-{np.std(accs):.3f}")
    os.makedirs(out_dir, exist_ok=True)
    pth = os.path.join(out_dir, "noise_resilience.csv")
    with open(pth, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", pth)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--m", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--p-deps", type=float, nargs="+",
                    default=[0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    ap.add_argument("--q-flips", type=float, nargs="+", default=[0.0, 0.03])
    ap.add_argument("--shots", type=int, default=500)
    ap.add_argument("--out", default="results/fisher_information")
    args = ap.parse_args(argv)
    run(args.N, args.m, args.seeds, args.p_deps, args.q_flips, args.out, shots=args.shots)


if __name__ == "__main__":
    main()
