"""
experiments/fisher_information/certificates/la_single_setting_estimator.py
==========================================================
LEG LA — single-setting, post-selection-free estimator of a cross-irrep coherence.

Claim (operational replacement for an earlier, non-operational
C = sum_{k!=k'} |v_k^dag M v_{k'}|, a convention-dependent sum-of-abs):

  For G = Z_N (shift rep U_g|x>=|x+g mod N>), base state phi, concept psi = U_s phi,
  the frame+system pre-measurement state is
        |Phi> = (1/sqrt(N)) sum_g |g>_R (x) U_g psi.
  Fourier frame basis |e_k> = sum_g F[g,k]|g>,  F[g,k] = omega^{gk}/sqrt(N), omega=e^{2pi i/N}.
  Frame components v_k = <e_k|Phi> = (1/N) sum_g omega^{-gk} U_g psi,  sum_k v_k v_k^dag = rho_sys.

  Fix a target irrep pair (k,k') and a system-only Hermitian observable M. Define the
  TWO Hermitian observables on frame (x) system

     R_Re = ( |e_k><e_k'| + |e_k'><e_k| ) / 2         O_Re = R_Re (x) M
     R_Im = ( |e_k><e_k'| - |e_k'><e_k| ) / (2i)      O_Im = R_Im (x) M

  Then  <Phi|O_Re|Phi> = Re( v_k^dag M v_{k'} )   and   <Phi|O_Im|Phi> = Im( v_k^dag M v_{k'} ).

  O_Re, O_Im are PRODUCT observables: measure the frame register in the fixed rotated-
  Fourier basis that diagonalises R_Re (resp. R_Im) and the system in the eigenbasis of M;
  multiply the two +/-outcomes. Every prepared |Phi> yields a sample (NO post-selection,
  no accept-1/N wall). The single-shot value Z = r*mu obeys |Z| <= ||O_Re|| = ||M||/2, so
  Var(Z) <= ||M||^2/4 <= ||M||^2, giving additive-eps precision at n = O(1/eps^2) shots,
  INDEPENDENT of N.

CERTIFICATE (this file):
  (1) exactness:  |<Phi|O|Phi> - analytic Re/Im(v_k^dag M v_{k'})| < 1e-12   (numeric==analytic)
  (2) unbiasedness at finite shots: |empirical mean - analytic| small (~1e-2) and ->0
  (3) variance bound: n * Var(sample-mean) = Var(Z) <= ||M||^2   (the O(1/eps^2) bound; PASS)
  over {Z4, Z8}, multiple observables/pairs/seeds.

Output CSV: results/fisher_information/certificates/la_single_setting_estimator.csv
Pure numpy, deterministic per seed.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np


# ---------------------------------------------------------------------------
# Z_N shift representation + Fourier frame
# ---------------------------------------------------------------------------
def shift(N: int, g: int) -> np.ndarray:
    return np.roll(np.eye(N, dtype=complex), g, axis=0)      # U_g|x> = |x+g mod N>


def fourier_frame(N: int) -> np.ndarray:
    """F[g,k] = omega^{gk}/sqrt(N), omega = e^{2pi i/N}  (unitary DFT frame)."""
    g = np.arange(N)
    return np.exp(2j * np.pi * np.outer(g, g) / N) / np.sqrt(N)


def frame_components(N: int, psi: np.ndarray, F: np.ndarray) -> np.ndarray:
    """v_k = (1/N) sum_g omega^{-gk} U_g psi = (1/sqrt N) sum_g conj(F[g,k]) U_g psi.
    Returns V with row k = v_k (shape (N, N))."""
    Upsi = np.stack([shift(N, g) @ psi for g in range(N)], axis=0)   # (N, N): row g = U_g psi
    return (F.conj().T @ Upsi) / np.sqrt(N)                          # (N, N): row k = v_k


def joint_state(N: int, psi: np.ndarray) -> np.ndarray:
    """|Phi> = (1/sqrt N) sum_g |g>_R (x) U_g psi,  length N*N, index g*N + i."""
    Phi = np.zeros(N * N, dtype=complex)
    for g in range(N):
        Phi[g * N:(g + 1) * N] = (shift(N, g) @ psi) / np.sqrt(N)
    return Phi


def random_state(d, rng):
    v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    return v / np.linalg.norm(v)


def random_hermitian(d, rng):
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    return (A + A.conj().T) / 2.0


# ---------------------------------------------------------------------------
# the estimator: exact expectation + finite-shot simulation
# ---------------------------------------------------------------------------
def observables_R(F, k, kp):
    """Hermitian frame operators R_Re, R_Im on the N-dim frame register."""
    ek = F[:, k]
    ekp = F[:, kp]
    outer_kkp = np.outer(ek, ekp.conj())          # |e_k><e_k'|
    R_Re = (outer_kkp + outer_kkp.conj().T) / 2.0
    R_Im = (outer_kkp - outer_kkp.conj().T) / (2j)
    return R_Re, R_Im


def exact_and_sample(N, psi, M, F, k, kp, shots, rng):
    """Return dict of exact expectations, analytic coherence, and finite-shot means/vars
    for both Re and Im single-setting product measurements on |Phi>."""
    V = frame_components(N, psi, F)
    coh = complex(V[k].conj() @ M @ V[kp])         # analytic v_k^dag M v_{k'}
    Phi = joint_state(N, psi)
    R_Re, R_Im = observables_R(F, k, kp)
    Mnorm = float(np.linalg.norm(M, 2))            # operator norm of M

    out = dict(coh_re=float(coh.real), coh_im=float(coh.imag), Mnorm=Mnorm)

    # system-side eigendecomposition (shared)
    mu, Meig = np.linalg.eigh(M)                    # mu[b], columns Meig[:,b]

    for tag, R in (("re", R_Re), ("im", R_Im)):
        # exact expectation <Phi| R (x) M |Phi>
        O = np.kron(R, M)
        exact = float(np.real(Phi.conj() @ O @ Phi))
        out[f"exact_{tag}"] = exact

        # finite-shot product measurement: frame in eigenbasis of R, system in eigenbasis of M
        r, Reig = np.linalg.eigh(R)                 # r[a], columns Reig[:,a]
        # amplitude A[a,b] = (f_a (x) m_b)^dag |Phi>
        # = Reig[:,a]^dag_R  m_b^dag_S  applied to Phi (reshaped N_R x N_S)
        Phi_mat = Phi.reshape(N, N)                 # [g_frame, i_system]
        Amp = Reig.conj().T @ Phi_mat @ Meig.conj() # (N_frame_out, N_sys_out)
        probs = np.abs(Amp) ** 2
        probs = probs / probs.sum()                 # guard normalisation (~1)
        eigval_grid = np.outer(r, mu)               # value r_a * mu_b for outcome (a,b)

        flat_p = probs.ravel()
        flat_v = eigval_grid.ravel()
        # exact single-shot moments from the distribution
        m1 = float(flat_p @ flat_v)
        m2 = float(flat_p @ (flat_v ** 2))
        var_shot = m2 - m1 ** 2
        out[f"exact_shotmean_{tag}"] = m1            # == exact (consistency)
        out[f"var_single_shot_{tag}"] = var_shot     # = n * Var(sample mean)

        # actual Monte-Carlo sampling of `shots` outcomes
        idx = rng.choice(flat_p.size, size=shots, p=flat_p)
        samp = flat_v[idx]
        out[f"emp_mean_{tag}"] = float(samp.mean())
        out[f"emp_abserr_{tag}"] = abs(float(samp.mean()) - exact)
    return out


# ---------------------------------------------------------------------------
# driver / certificate
# ---------------------------------------------------------------------------
def run(sizes, n_obs, shots, seeds, out_path):
    rows = []
    worst_exact_err = 0.0
    worst_var_over_M2 = 0.0
    worst_emp_err = 0.0
    for N in sizes:
        F = fourier_frame(N)
        # a couple of distinct cross-irrep pairs (k != k')
        pairs = [(1, 2), (0, 1), (1, N - 1)]
        for sd in seeds:
            rng = np.random.default_rng(4000 + sd)
            for _ in range(n_obs):
                # concept psi = U_s phi with random base phi and random hidden shift s
                phi = random_state(N, rng)
                s = int(rng.integers(0, N))
                psi = shift(N, s) @ phi
                M = random_hermitian(N, rng)
                for (k, kp) in pairs:
                    d = exact_and_sample(N, psi, M, F, k, kp, shots, rng)
                    Mnorm = d["Mnorm"]
                    for tag in ("re", "im"):
                        exact = d[f"exact_{tag}"]
                        coh = d["coh_re"] if tag == "re" else d["coh_im"]
                        exact_err = abs(exact - coh)                 # numeric==analytic
                        # consistency: exact expectation vs distribution mean
                        shotmean_err = abs(d[f"exact_shotmean_{tag}"] - exact)
                        var_shot = d[f"var_single_shot_{tag}"]
                        var_over_M2 = var_shot / (Mnorm ** 2 + 1e-300)
                        emp_err = d[f"emp_abserr_{tag}"]
                        worst_exact_err = max(worst_exact_err, exact_err, shotmean_err)
                        worst_var_over_M2 = max(worst_var_over_M2, var_over_M2)
                        worst_emp_err = max(worst_emp_err, emp_err)
                        rows.append(dict(
                            group=f"Z{N}", N=N, k=k, kp=kp, part=tag, seed=sd,
                            analytic_coh=coh, exact_expect=exact,
                            exact_vs_analytic_err=exact_err,
                            shotmean_vs_exact_err=shotmean_err,
                            Mnorm=Mnorm, var_single_shot=var_shot,
                            var_over_Mnorm2=var_over_M2,
                            bound_quarter_ok=int(var_shot <= 0.25 * Mnorm ** 2 + 1e-12),
                            bound_full_ok=int(var_shot <= Mnorm ** 2 + 1e-12),
                            shots=shots, emp_mean=d[f"emp_mean_{tag}"],
                            emp_abs_err=emp_err,
                        ))
        # progress print
        sub = [r for r in rows if r["N"] == N]
        me = max(r["exact_vs_analytic_err"] for r in sub)
        mv = max(r["var_over_Mnorm2"] for r in sub)
        mee = max(r["emp_abs_err"] for r in sub)
        print(f"  Z{N}: max|exact-analytic|={me:.2e}  max Var/||M||^2={mv:.3f}  "
              f"max|emp-exact|@{shots}shots={mee:.3e}  ({len(sub)} rows)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", out_path)
    print("=" * 70)
    print("CERTIFICATE SUMMARY  (LEG LA)")
    print("=" * 70)
    print(f"  (1) exactness   : max|<Phi|O|Phi> - analytic Re/Im| = {worst_exact_err:.2e}"
          f"   (threshold 1e-12 -> {'PASS' if worst_exact_err < 1e-12 else 'FAIL'})")
    print(f"  (2) unbiasedness: max|emp mean - exact| @ {shots} shots = {worst_emp_err:.3e}"
          f"   (finite-shot, ~1e-2 regime)")
    print(f"  (3) variance    : max Var_single_shot/||M||^2 = {worst_var_over_M2:.4f}"
          f"   (bound <=1 -> {'PASS' if worst_var_over_M2 <= 1.0 + 1e-9 else 'FAIL'};"
          f" tight bound 1/4={'ok' if worst_var_over_M2 <= 0.25 + 1e-9 else 'exceeds 1/4'})")
    return rows, worst_exact_err, worst_var_over_M2, worst_emp_err


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--n_obs", type=int, default=6)
    ap.add_argument("--shots", type=int, default=200000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/fisher_information/certificates/la_single_setting_estimator.csv")
    args = ap.parse_args(argv)
    run(args.sizes, args.n_obs, args.shots, args.seeds, args.out)


if __name__ == "__main__":
    main()
