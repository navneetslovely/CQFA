"""
LEG LD -- Access iff-superposition characterization of cross-irrep phases.
==========================================================================
Replaces the FALSE "Mode F is the unique phase-accessing readout" with the
correct linear-algebra iff.

SETUP (G = Z_N, shift/regular rep U_g|x>=|x+g mod N>, base phi, concept
psi_s = U_s phi):

    |Phi_s> = N^{-1/2} sum_g |g>_R (x) U_g psi_s .

KEY STRUCTURAL FACT (proved, exact).  Let |k^> = N^{-1/2} sum_x w^{kx}|x>,
w=exp(2pi i/N), be the system irrep vectors (U_g|k^> = w^{-kg}|k^>), and
|k~>_R = N^{-1/2} sum_g w^{-kg}|g>_R the frame Fourier vectors.  Then the joint
state has a PERFECT R-S Schmidt correlation between frame Fourier mode k and
system irrep k:

    |Phi_s> = sum_k c_k(s) |k~>_R (x) |k^>_S ,   c_k(s) = w^{-ks} <k^|phi> .

The "Fourier vector k" of the concept is v_k = P_k psi_s = c_k(s)|k^>.
The cross-irrep phase is  theta_{kk'}(s) = arg c_k(s) - arg c_{k'}(s)
                                       = -2pi (k-k') s / N + const.

READOUT.  A frame-register orthonormal readout basis {|b>_R} (columns of a
unitary F on R) with a subsequent system observable M realises the joint
observable O_b = |b><b|_R (x) M.  Write beta_{bk} = <k~|b>_R (the overlap of the
readout vector with frame Fourier mode k; by the Schmidt correlation this equals
the overlap of the outcome-b conditional system component with the Fourier vector
v_k).  The outcome statistic is
    S_b(s) = <Phi_s| O_b |Phi_s>
           = sum_{k,k'} beta_{bk} conj(beta_{bk'}) conj(c_k) c_{k'} M_{k'k}   [M_{k'k}=<k'^|M|k^>]
The theta_{kk'} dependence lives ONLY in the k!=k' cross terms, with coefficient
    beta_{bk} conj(beta_{bk'}) M_{k'k}.

THEOREM (LD).  d S_b / d theta_{kk'} can be made nonzero (over choice of system
observable M) IFF beta_{bk} != 0 AND beta_{bk'} != 0, i.e. the readout vector |b>
SUPERPOSES frame Fourier modes k and k' (equivalently its conditional system
component overlaps BOTH Fourier vectors v_k and v_{k'}).  In matrix form: writing
the readout unitary in the Fourier basis, B = F expressed with rows = outcomes,
columns = Fourier modes (B_{bk}=beta_{bk}), the readout accesses theta_{kk'} iff
SOME ROW of B has nonzero entries in BOTH columns k and k'.

COROLLARIES.
 * Fourier-basis frame readout (B = permutation/monomial in the Fourier basis):
   every row has a single nonzero -> superposes NO pair -> per-outcome phase-blind.
 * Mode B (classical FA = trace out the frame = observable I_R (x) M): O has
   O_{(k),(k')} ~ <k~|k'~> = delta_{kk'} -> phase-blind (recovers LB structurally).
 * A superposing readout (Haar-random F, or the interference pair
   |k~>+-|k'~>) accesses the phase.

This script CERTIFIES the iff numerically on {Z4, Z8}:
  (A) Schmidt reconstruction: || |Phi_s> - sum_k c_k(s)|k~>|k^> || < 1e-12.
  (B) numeric == analytic: the measured theta-amplitude of S_b (half its exact
      peak-to-peak over a theta grid, a pure outcome-statistic with no finite
      differencing) equals the closed form  2 (r_k r_kp / ||.||^2)|beta_{bk}||beta_{bk'}|
      to ~1e-12.
  (C) iff: predicted-accessible (both Fourier overlaps > tol) == measured-accessible
      (theta-amplitude of S_b > tol) for EVERY (readout-basis, outcome, pair), across
      random readout unitaries + Fourier + group(=identity) bases.  0 mismatches.
  (D) diagonal (Fourier-basis) readout: per-outcome theta-amplitude < 1e-12.
  (E) Mode B observable I_R (x) M: theta-amplitude < 1e-12.

Every reported number is written to results/fisher_information/certificates/ld_access_iff_rank.csv .
"""
from __future__ import annotations

import csv
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO, "results", "fisher_information", "certificates")


def shift(N: int, g: int) -> np.ndarray:
    """U_g |x> = |x+g mod N>  (cyclic shift / regular rep of Z_N)."""
    return np.roll(np.eye(N, dtype=complex), g, axis=0)


def irrep_vec(N: int, k: int) -> np.ndarray:
    """|k^> = N^{-1/2} sum_x w^{kx}|x>, w=exp(2pi i/N).  U_g|k^> = w^{-kg}|k^>."""
    x = np.arange(N)
    return np.exp(2j * np.pi * k * x / N) / np.sqrt(N)


def frame_fourier_vec(N: int, k: int) -> np.ndarray:
    """|k~>_R = N^{-1/2} sum_g w^{-kg}|g>_R."""
    g = np.arange(N)
    return np.exp(-2j * np.pi * k * g / N) / np.sqrt(N)


def joint_state(N: int, psi: np.ndarray) -> np.ndarray:
    """|Phi> = N^{-1/2} sum_g |g>_R (x) U_g psi  in C^{N*N} (R outer, S inner)."""
    out = np.zeros(N * N, dtype=complex)
    for g in range(N):
        Ug = shift(N, g)
        out[g * N:(g + 1) * N] = Ug @ psi
    return out / np.sqrt(N)


def random_unitary(n, rng):
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def concept_two_mode(N, k, kp, r_k, r_kp, theta):
    """psi = r_k |k^> + r_kp e^{i theta} |kp^>  (two-irrep concept, unit norm).
    The relative phase theta = theta_{kk'} is the cross-irrep phase to be probed."""
    psi = r_k * irrep_vec(N, k) + r_kp * np.exp(1j * theta) * irrep_vec(N, kp)
    return psi / np.linalg.norm(psi)


def outcome_stat(N, psi, b_R, M):
    """S_b = <Phi|(|b><b|_R (x) M)|Phi>, with |Phi> built from psi and readout
    frame vector b_R (length N).  Realised on the joint state directly."""
    Phi = joint_state(N, psi)                       # (N*N,)
    Phi = Phi.reshape(N, N)                          # [g, x]  = R (x) S blocks
    # conditional (unnormalised) system component w_b = <b|_R |Phi>
    w_b = b_R.conj() @ Phi                           # (N,)  sum_g conj(b_R[g]) Phi[g,:]
    return float(np.real(w_b.conj() @ M @ w_b))


def modeB_stat(N, psi, M):
    """Mode B observable I_R (x) M: S = <Phi|(I_R (x) M)|Phi> = Tr(M rho_sys)."""
    Phi = joint_state(N, psi).reshape(N, N)
    val = 0.0
    for g in range(N):
        w = Phi[g]
        val += np.real(w.conj() @ M @ w)
    return float(val)


def main():
    groups = [4, 8]
    n_pairs = 6          # random (k,k') pairs per group
    n_readouts = 4       # random readout unitaries per group
    tol = 1e-9           # accessibility threshold on the theta-amplitude of S_b
    n_grid = 256         # theta grid for the exact peak-to-peak amplitude

    rows = []
    global_max_reconstruct = 0.0
    global_max_numvsana = 0.0
    global_iff_mismatches = 0
    global_iff_total = 0
    global_max_diag_blind = 0.0
    global_max_modeB_blind = 0.0
    global_min_access_deriv = np.inf   # smallest nonzero derivative among truly-accessible

    print("LEG LD -- access iff-superposition certificate")
    print("=" * 70)

    for N in groups:
        rng = np.random.default_rng(3000 + N)

        # (A) Schmidt reconstruction over random phi, s
        max_reconstruct = 0.0
        for _ in range(20):
            v = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            phi = v / np.linalg.norm(v)
            for s in range(N):
                psi = shift(N, s) @ phi
                Phi = joint_state(N, psi)
                # analytic: sum_k c_k |k~>(x)|k^>, c_k = w^{-ks}<k^|phi>
                rec = np.zeros(N * N, dtype=complex)
                for k in range(N):
                    ck = np.exp(-2j * np.pi * k * s / N) * (irrep_vec(N, k).conj() @ phi)
                    rec += ck * np.kron(frame_fourier_vec(N, k), irrep_vec(N, k))
                max_reconstruct = max(max_reconstruct,
                                      float(np.linalg.norm(Phi - rec)))
        global_max_reconstruct = max(global_max_reconstruct, max_reconstruct)

        # readout bases to test: random unitaries + Fourier(DFT) + group(identity)
        readouts = []
        for j in range(n_readouts):
            readouts.append((f"random{j}", random_unitary(N, rng)))
        # Fourier frame basis columns = |k~>_R  (this is the "diagonal in Fourier" readout)
        Ffour = np.stack([frame_fourier_vec(N, k) for k in range(N)], axis=1)  # cols
        readouts.append(("fourier", Ffour))
        readouts.append(("group", np.eye(N, dtype=complex)))                   # |g>_R basis

        max_numvsana = 0.0
        iff_mismatch = 0
        iff_total = 0
        max_diag_blind = 0.0
        max_modeB_blind = 0.0
        min_access_deriv = np.inf

        for pidx in range(n_pairs):
            k, kp = rng.choice(N, size=2, replace=False)
            k, kp = int(k), int(kp)
            r_k = float(rng.uniform(0.4, 1.0))
            r_kp = float(rng.uniform(0.4, 1.0))
            theta0 = float(rng.uniform(0, 2 * np.pi))
            # system observable connecting irreps k,k' :  M = |k^><kp^| + |kp^><k^|
            vk, vkp = irrep_vec(N, k), irrep_vec(N, kp)
            M = np.outer(vk, vkp.conj()) + np.outer(vkp, vk.conj())
            Mkkp = 1.0  # <k^|M|kp^> = 1 by construction

            for rname, F in readouts:
                for b in range(N):
                    b_R = F[:, b]
                    # overlaps with frame Fourier modes k, k'  (rows of B in Fourier basis)
                    beta_k = frame_fourier_vec(N, k).conj() @ b_R
                    beta_kp = frame_fourier_vec(N, kp).conj() @ b_R
                    superposes = (abs(beta_k) > 1e-10) and (abs(beta_kp) > 1e-10)

                    # measured theta-amplitude of the outcome statistic S_b(theta).
                    # S_b(theta) = a0 + 2C Re(z e^{i th}) is DC + a single harmonic, so
                    # its amplitude is extracted EXACTLY from the order-1 DFT coefficient
                    # over a uniform theta grid (a pure linear functional of the measured
                    # outcome statistics; a constant -> machine-zero amplitude).
                    grid = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
                    Svals = np.array([outcome_stat(
                        N, concept_two_mode(N, k, kp, r_k, r_kp, th), b_R, M)
                        for th in grid])
                    c1 = np.mean(Svals * np.exp(-1j * grid))
                    meas_amp = float(2.0 * abs(c1))

                    # closed form: S_b(theta) = 2 (r_k r_kp/norm^2) Re(beta_k conj(beta_kp) e^{i th})
                    # -> theta-amplitude = 2 (r_k r_kp/norm^2) |beta_k||beta_kp| * |M_{kp k}|.
                    nrm2 = r_k ** 2 + r_kp ** 2
                    ana_amp = 2 * abs(beta_k) * abs(beta_kp) * (r_k * r_kp / nrm2) * Mkkp
                    max_numvsana = max(max_numvsana, abs(meas_amp - ana_amp))

                    measured_access = meas_amp > tol
                    iff_total += 1
                    if measured_access != superposes:
                        iff_mismatch += 1
                    if superposes:
                        min_access_deriv = min(min_access_deriv, meas_amp)
                    # (D) diagonal (fourier) readout blindness bookkeeping
                    if rname == "fourier" and not superposes:
                        max_diag_blind = max(max_diag_blind, meas_amp)

                # (E) Mode B observable I_R (x) M  : blind for this pair
                gridB = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
                SBvals = np.array([modeB_stat(
                    N, concept_two_mode(N, k, kp, r_k, r_kp, th), M) for th in gridB])
                max_modeB_blind = max(max_modeB_blind,
                                      float((SBvals.max() - SBvals.min()) / 2.0))

        rows.append(dict(
            group=f"Z{N}", N=N, n_pairs=n_pairs, n_readouts=len(readouts),
            iff_total=iff_total, iff_mismatches=iff_mismatch,
            max_schmidt_reconstruct=max_reconstruct,
            max_numeric_vs_analytic=max_numvsana,
            max_fourier_diag_blind_deriv=max_diag_blind,
            max_modeB_blind_deriv=max_modeB_blind,
            min_accessible_deriv=(0.0 if min_access_deriv is np.inf else min_access_deriv),
        ))
        global_max_numvsana = max(global_max_numvsana, max_numvsana)
        global_iff_mismatches += iff_mismatch
        global_iff_total += iff_total
        global_max_diag_blind = max(global_max_diag_blind, max_diag_blind)
        global_max_modeB_blind = max(global_max_modeB_blind, max_modeB_blind)
        global_min_access_deriv = min(global_min_access_deriv, min_access_deriv)

        print(f"Z{N:<3} | reconstruct<= {max_reconstruct:.2e} | num-vs-ana<= "
              f"{max_numvsana:.2e} | iff mism {iff_mismatch}/{iff_total} | "
              f"fourier-diag blind<= {max_diag_blind:.2e} | modeB blind<= "
              f"{max_modeB_blind:.2e} | min access deriv {min_access_deriv:.2e}")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "ld_access_iff_rank.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print("=" * 70)
    print(f"GLOBAL: schmidt-reconstruct<= {global_max_reconstruct:.2e}  "
          f"num-vs-analytic<= {global_max_numvsana:.2e}")
    print(f"        iff mismatches = {global_iff_mismatches}/{global_iff_total}  "
          f"fourier-diag blind<= {global_max_diag_blind:.2e}  "
          f"modeB blind<= {global_max_modeB_blind:.2e}")
    print(f"        min accessible derivative (separation) = {global_min_access_deriv:.3e}")
    checks = {
        "schmidt_reconstruct<1e-12": global_max_reconstruct < 1e-12,
        "num_vs_analytic<1e-10": global_max_numvsana < 1e-10,
        "iff_zero_mismatches": global_iff_mismatches == 0,
        "fourier_diag_blind<1e-12": global_max_diag_blind < 1e-12,
        "modeB_blind<1e-12": global_max_modeB_blind < 1e-12,
        "accessible_separated>1e-3": global_min_access_deriv > 1e-3,
    }
    print("CERTIFICATE:", "PASS" if all(checks.values()) else "FAIL", checks)
    print("wrote", path)


if __name__ == "__main__":
    main()
