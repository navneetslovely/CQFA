"""
LEG LD -- Access iff-rank (matrix-form) certificate.
====================================================
Task-named deliverable for LEG LD.  This CERTIFIES the LD theorem in its
MATRIX form -- the level at which it replaces the false "uniqueness" claim --
and adds the reference-interference-pair corollary that the per-outcome cert
(ld_access_iff_rank.py) does not isolate.

SETUP (G = Z_N, shift/regular rep U_g|x>=|x+g mod N>, base phi, concept
psi_s = U_s phi):
    |Phi_s> = N^{-1/2} sum_g |g>_R (x) U_g psi_s .

STEP-1 fact (proved, exact).  With |k^>=N^{-1/2} sum_x w^{kx}|x> (w=e^{2pi i/N},
U_g|k^>=w^{-kg}|k^>) and |k~>_R=N^{-1/2} sum_g w^{-kg}|g>_R,
    |Phi_s> = sum_k c_k(s) |k~>_R (x) |k^>_S ,   c_k(s) = w^{-ks}<k^|phi> ,
a PERFECT Schmidt correlation (frame Fourier mode k <-> system irrep k).
Cross-irrep phase  theta_{kk'}(s) = arg c_k - arg c_{k'} = -2pi(k-k')s/N + const.

READOUT.  Frame orthonormal readout basis = columns of a unitary F on R; write
its matrix in the frame-Fourier basis  B = W_dag F  where W has columns |k~>_R,
so B_{bk} = beta_{bk} = <k~|b>_R.  Outcome b then a system observable M gives
    S_b(s) = <Phi_s|(|b><b|_R (x) M)|Phi_s>
           = sum_{k,k'} beta_{bk} conj(beta_{bk'}) conj(c_k) c_{k'} <k'^|M|k^>,
and the theta_{kk'} dependence has coefficient beta_{bk} conj(beta_{bk'}) M_{k'k}.

THEOREM (LD, exact).  With the witness M = |k^><k'^| + |k'^><k^| and a concept
supported on {k,k'} (psi = r_k|k^> + r_{k'} e^{i theta}|k'^>), the outcome
statistic is the closed form
    S_b(theta) = 2 (r_k r_{k'}/(r_k^2+r_{k'}^2)) Re( beta_{bk} conj(beta_{bk'}) e^{i theta} )
whose theta-amplitude is  2 (r_k r_{k'}/(r_k^2+r_{k'}^2)) |beta_{bk}||beta_{bk'}|.
Hence outcome b accesses theta_{kk'} IFF beta_{bk}!=0 AND beta_{bk'}!=0.

MATRIX FORM (the object that replaces "uniqueness").  A whole readout basis F
accesses theta_{kk'} (via SOME outcome b) IFF SOME ROW of B has nonzero entries
in BOTH columns k and k'.  Corollaries:
  (a) Mode B / classical FA = observable I_R (x) M (trace out R): phase-blind
      (recovers LB) -- <k~|k'~>=delta_{kk'}.
  (b) Fourier-basis frame readout (F = W, so B = I): every row single-mode ->
      superposes NO pair -> per-outcome phase-blind.
  (c) reference-interference pair |k~>+-|k'~> (one row with both columns k,k'
      nonzero) DOES access theta_{kk'} -- the Mode-F interference primitive.

CERTIFIED here on {Z4, Z8, Z16}, all numbers written to ld_cert.csv:
  (1) Schmidt reconstruct  ||Phi - sum_k c_k|k~>|k^>||               < 1e-12
  (2) numeric == analytic theta-amplitude of S_b (exact 1-harmonic)  < 1e-12
  (3) MATRIX-FORM iff: for EVERY (readout basis F, pair k,k') the matrix
      predictor (some row of B nonzero in both cols) == the measured predictor
      (some outcome b has theta-amplitude > tol).  0 mismatches.
  (4) Mode B (I_R (x) M) theta-amplitude                             < 1e-12
  (5) Fourier-basis readout, per-outcome theta-amplitude             < 1e-12
  (6) interference pair |k~>+-|k'~> theta-amplitude == r_k r_kp/norm2 (closed
      form; strictly > 0 separated).
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
    """|k^> = N^{-1/2} sum_x w^{kx}|x>,  U_g|k^> = w^{-kg}|k^>."""
    x = np.arange(N)
    return np.exp(2j * np.pi * k * x / N) / np.sqrt(N)


def frame_fourier_vec(N: int, k: int) -> np.ndarray:
    """|k~>_R = N^{-1/2} sum_g w^{-kg}|g>_R."""
    g = np.arange(N)
    return np.exp(-2j * np.pi * k * g / N) / np.sqrt(N)


def W_matrix(N: int) -> np.ndarray:
    """Columns = frame-Fourier vectors |k~>_R; unitary. B = W_dag F."""
    return np.stack([frame_fourier_vec(N, k) for k in range(N)], axis=1)


def joint_state(N: int, psi: np.ndarray) -> np.ndarray:
    """|Phi> = N^{-1/2} sum_g |g>_R (x) U_g psi, reshaped [g, x]."""
    out = np.zeros((N, N), dtype=complex)
    for g in range(N):
        out[g] = shift(N, g) @ psi
    return out / np.sqrt(N)


def random_unitary(n, rng):
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def concept_two_mode(N, k, kp, r_k, r_kp, theta):
    """psi = (r_k|k^> + r_kp e^{i theta}|kp^>) normalized. Relative phase = theta_{kk'}."""
    psi = r_k * irrep_vec(N, k) + r_kp * np.exp(1j * theta) * irrep_vec(N, kp)
    return psi / np.linalg.norm(psi)


def outcome_stat(N, psi, b_R, M):
    """S_b = <Phi|(|b><b|_R (x) M)|Phi>. b_R = readout frame vector (length N)."""
    Phi = joint_state(N, psi)            # [g, x]
    w_b = b_R.conj() @ Phi               # conditional (unnormalised) system component
    return float(np.real(w_b.conj() @ M @ w_b))


def modeB_stat(N, psi, M):
    """Mode B observable I_R (x) M: S = sum_g <w_g|M|w_g> = Tr(M rho_sys)."""
    Phi = joint_state(N, psi)
    return float(sum(np.real(Phi[g].conj() @ M @ Phi[g]) for g in range(N)))


def theta_amplitude(Svals, grid):
    """S(theta)=DC + single harmonic; amplitude = 2|order-1 DFT coeff| (exact)."""
    return float(2.0 * abs(np.mean(Svals * np.exp(-1j * grid))))


def main():
    groups = [4, 8, 16]
    n_readouts = 4       # random readout unitaries per group
    tol = 1e-9           # accessibility threshold on theta-amplitude
    zero_tol = 1e-10     # nonzero-entry threshold for beta_{bk}
    n_grid = 64          # S(theta) is DC + one harmonic; >2 samples suffice (exact DFT)

    rows = []
    g_reconstruct = 0.0
    g_numana = 0.0
    g_matrix_mism = 0
    g_matrix_total = 0
    g_modeB_blind = 0.0
    g_fourier_blind = 0.0
    g_interf_err = 0.0
    g_min_access = np.inf

    print("LEG LD -- access iff-rank (matrix-form) certificate")
    print("=" * 72)

    for N in groups:
        rng = np.random.default_rng(7000 + N)
        W = W_matrix(N)
        grid = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)

        # (1) Schmidt reconstruction, random phi and all shifts s
        max_reconstruct = 0.0
        for _ in range(15):
            v = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            phi = v / np.linalg.norm(v)
            for s in range(N):
                psi = shift(N, s) @ phi
                Phi = joint_state(N, psi).reshape(-1)
                rec = np.zeros(N * N, dtype=complex)
                for k in range(N):
                    ck = np.exp(-2j * np.pi * k * s / N) * (irrep_vec(N, k).conj() @ phi)
                    rec += ck * np.kron(frame_fourier_vec(N, k), irrep_vec(N, k))
                max_reconstruct = max(max_reconstruct, float(np.linalg.norm(Phi - rec)))

        # readout bases: random unitaries + Fourier (F=W, B=I) + group (F=I)
        readouts = [(f"random{j}", random_unitary(N, rng)) for j in range(n_readouts)]
        readouts.append(("fourier", W.copy()))
        readouts.append(("group", np.eye(N, dtype=complex)))

        max_numana = 0.0
        matrix_mism = 0
        matrix_total = 0
        max_modeB = 0.0
        max_fourier_blind = 0.0
        max_interf_err = 0.0
        min_access = np.inf

        # all unordered irrep pairs
        pairs = [(k, kp) for k in range(N) for kp in range(k + 1, N)]

        for (k, kp) in pairs:
            r_k = float(rng.uniform(0.4, 1.0))
            r_kp = float(rng.uniform(0.4, 1.0))
            nrm2 = r_k ** 2 + r_kp ** 2
            vk, vkp = irrep_vec(N, k), irrep_vec(N, kp)
            M = np.outer(vk, vkp.conj()) + np.outer(vkp, vk.conj())   # <k^|M|kp^>=1

            for rname, F in readouts:
                B = W.conj().T @ F                # readout matrix in Fourier basis, cols=modes
                # MATRIX predictor: SOME row b of B has both |B[b,k]|,|B[b,kp]| nonzero
                matrix_access = bool(np.any(
                    (np.abs(B[:, k]) > zero_tol) & (np.abs(B[:, kp]) > zero_tol)))

                # MEASURED predictor: SOME outcome b has theta-amplitude > tol
                measured_any = False
                for b in range(N):
                    b_R = F[:, b]
                    beta_k = frame_fourier_vec(N, k).conj() @ b_R
                    beta_kp = frame_fourier_vec(N, kp).conj() @ b_R
                    Svals = np.array([outcome_stat(
                        N, concept_two_mode(N, k, kp, r_k, r_kp, th), b_R, M) for th in grid])
                    meas_amp = theta_amplitude(Svals, grid)
                    ana_amp = 2 * abs(beta_k) * abs(beta_kp) * (r_k * r_kp / nrm2)
                    max_numana = max(max_numana, abs(meas_amp - ana_amp))
                    if meas_amp > tol:
                        measured_any = True
                        min_access = min(min_access, meas_amp)
                    if rname == "fourier":
                        # per-outcome Fourier readout must be phase-blind
                        max_fourier_blind = max(max_fourier_blind, meas_amp)

                matrix_total += 1
                if matrix_access != measured_any:
                    matrix_mism += 1

            # (4) Mode B I_R (x) M phase-blind for this pair
            SB = np.array([modeB_stat(
                N, concept_two_mode(N, k, kp, r_k, r_kp, th), M) for th in grid])
            max_modeB = max(max_modeB, theta_amplitude(SB, grid))

            # (6) reference-interference pair |k~>+-|k'~> accesses; closed form r_k r_kp/nrm2
            for sign in (+1.0, -1.0):
                b_R = (frame_fourier_vec(N, k) + sign * frame_fourier_vec(N, kp)) / np.sqrt(2)
                Svals = np.array([outcome_stat(
                    N, concept_two_mode(N, k, kp, r_k, r_kp, th), b_R, M) for th in grid])
                meas_amp = theta_amplitude(Svals, grid)
                # beta_k=beta_kp=1/sqrt2 -> amp = 2*(1/2)*(r_k r_kp/nrm2) = r_k r_kp/nrm2
                interf_ana = r_k * r_kp / nrm2
                max_interf_err = max(max_interf_err, abs(meas_amp - interf_ana))

        rows.append(dict(
            group=f"Z{N}", N=N, n_pairs=len(pairs), n_readouts=len(readouts),
            matrix_iff_total=matrix_total, matrix_iff_mismatches=matrix_mism,
            max_schmidt_reconstruct=max_reconstruct,
            max_numeric_vs_analytic=max_numana,
            max_modeB_blind_amp=max_modeB,
            max_fourier_readout_blind_amp=max_fourier_blind,
            max_interference_pair_err=max_interf_err,
            min_accessible_amp=(0.0 if min_access is np.inf else min_access),
            max_blind_amp=max(max_modeB, max_fourier_blind, max_reconstruct),
        ))
        g_reconstruct = max(g_reconstruct, max_reconstruct)
        g_numana = max(g_numana, max_numana)
        g_matrix_mism += matrix_mism
        g_matrix_total += matrix_total
        g_modeB_blind = max(g_modeB_blind, max_modeB)
        g_fourier_blind = max(g_fourier_blind, max_fourier_blind)
        g_interf_err = max(g_interf_err, max_interf_err)
        g_min_access = min(g_min_access, min_access)

        print(f"Z{N:<3} | reconstruct<= {max_reconstruct:.2e} | num-vs-ana<= {max_numana:.2e} "
              f"| MATRIX-iff mism {matrix_mism}/{matrix_total} | modeB blind<= {max_modeB:.2e} "
              f"| fourier blind<= {max_fourier_blind:.2e} | interf err<= {max_interf_err:.2e} "
              f"| min access {min_access:.2e}")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "ld_cert.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print("=" * 72)
    print(f"GLOBAL: schmidt-reconstruct<= {g_reconstruct:.2e}  num-vs-analytic<= {g_numana:.2e}")
    print(f"        MATRIX-FORM iff mismatches = {g_matrix_mism}/{g_matrix_total}")
    print(f"        modeB blind<= {g_modeB_blind:.2e}  fourier-readout blind<= {g_fourier_blind:.2e}"
          f"  interference-pair err<= {g_interf_err:.2e}")
    # Separation = smallest genuinely-accessible amplitude vs largest blind amplitude.
    # The iff is unambiguous as long as these two populations do not overlap; the ratio
    # (not any fixed absolute floor) is the honest separation figure. A random readout row
    # can legitimately have both overlaps nonzero yet small, so min_access shrinks with N,
    # but it stays many orders above the blind band (all <= ~1e-14) and above tol=1e-9.
    max_blind = max(g_modeB_blind, g_fourier_blind, g_reconstruct)
    separation_ratio = (g_min_access / max_blind) if max_blind > 0 else np.inf
    print(f"        min accessible amplitude = {g_min_access:.3e}  max blind amplitude = "
          f"{max_blind:.3e}  separation ratio = {separation_ratio:.2e}")
    checks = {
        "schmidt_reconstruct<1e-12": g_reconstruct < 1e-12,
        "num_vs_analytic<1e-12": g_numana < 1e-12,
        "matrix_iff_zero_mismatches": g_matrix_mism == 0,
        "modeB_blind<1e-12": g_modeB_blind < 1e-12,
        "fourier_readout_blind<1e-12": g_fourier_blind < 1e-12,
        "interference_pair_matches<1e-12": g_interf_err < 1e-12,
        "accessible_above_tol": g_min_access > 1e3 * tol,   # accessible band >> 1e-9 tol
        "accessible_blind_separated": separation_ratio > 1e6,  # no overlap of bands
    }
    ok = all(bool(v) for v in checks.values())
    print("CERTIFICATE:", "PASS" if ok else "FAIL", checks)
    print("wrote", path)


if __name__ == "__main__":
    main()
