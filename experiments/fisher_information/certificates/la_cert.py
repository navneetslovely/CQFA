"""
experiments/fisher_information/certificates/la_cert.py
======================================
LEG LA certificate — single-setting, post-selection-free estimator of a
cross-irrep coherence.  Verifies, entirely from code + analytic closed forms,
every leg of the THEOREM (LA):

  Setup: G = Z_N, shift rep U_g|x>=|x+g mod N>, concept psi = U_s phi, joint
  frame+system state |Phi> = (1/sqrt N) sum_g |g>_R (x) U_g psi.  Fourier frame
  |e_k> = sum_g F[g,k]|g>, F[g,k]=omega^{gk}/sqrt N (omega=e^{2 pi i/N}); frame
  components v_k = <e_k|Phi> = (1/N) sum_g omega^{-gk} U_g psi, with
  sum_k v_k v_k^dag = rho_sys.  Fix a cross-irrep pair (k,k'), k!=k', and a
  system-only Hermitian M.  Define on frame (x) system
       R_Re = (|e_k><e_k'| + |e_k'><e_k|)/2,    O_Re = R_Re (x) M
       R_Im = (|e_k><e_k'| - |e_k'><e_k|)/(2i),  O_Im = R_Im (x) M

  (a) EXACTNESS:  <Phi|O_Re|Phi> = Re(v_k^dag M v_{k'}),
                  <Phi|O_Im|Phi> = Im(v_k^dag M v_{k'})  exactly.
  (b) SINGLE-SETTING, POST-SELECTION-FREE:  O_Re,O_Im are product observables
      (R on frame (x) M on system); each measured in ONE fixed non-adaptive
      basis.  R has spectrum {-1/2, 0, +1/2}; the 0-eigenspace (frame outcomes
      outside {e_k,e_k'}) is RECORDED as value 0, never discarded — every shot
      is used (probabilities sum to exactly 1; no accept-1/N wall).
  (c) VARIANCE BOUND / O(1/eps^2):  single-shot Z=r*mu obeys |Z| <= ||M||/2, so
      Var(Z) <= ||M||^2/4.  Hence additive std-error eps at n = ||M||^2/(4 eps^2)
      shots, INDEPENDENT of N.  Also |v_k^dag M v_{k'}|^2 = Re^2 + Im^2 is
      recovered from the two single-setting estimators.

Sanity: numeric == analytic to ~1e-12 for (a), the resolution-of-identity
sum_k v_k v_k^dag = rho_sys, and the finite-shot mean -> exact expectation.

Groups: Z4, Z8, Z16 and a prime-order (Z_p, p=7) instance.  Seconds to run.
Output CSV: results/fisher_information/certificates/la_cert.csv
Pure numpy, deterministic per seed.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np


# ---------------------------------------------------------------------------
# Z_N shift representation + Fourier frame + joint state
# ---------------------------------------------------------------------------
def shift(N: int, g: int) -> np.ndarray:
    return np.roll(np.eye(N, dtype=complex), g, axis=0)      # U_g|x> = |x+g mod N>


def fourier_frame(N: int) -> np.ndarray:
    """F[g,k] = omega^{gk}/sqrt(N), omega = e^{2 pi i/N}  (unitary DFT frame)."""
    g = np.arange(N)
    return np.exp(2j * np.pi * np.outer(g, g) / N) / np.sqrt(N)


def frame_components(N: int, psi: np.ndarray, F: np.ndarray) -> np.ndarray:
    """v_k = (1/N) sum_g omega^{-gk} U_g psi = (1/sqrt N) sum_g conj(F[g,k]) U_g psi.
    Returns V with row k = v_k (shape (N, N))."""
    Upsi = np.stack([shift(N, g) @ psi for g in range(N)], axis=0)   # row g = U_g psi
    return (F.conj().T @ Upsi) / np.sqrt(N)                          # row k = v_k


def joint_state(N: int, psi: np.ndarray) -> np.ndarray:
    """|Phi> = (1/sqrt N) sum_g |g>_R (x) U_g psi,  length N*N, index g*N + i."""
    Phi = np.zeros(N * N, dtype=complex)
    for g in range(N):
        Phi[g * N:(g + 1) * N] = (shift(N, g) @ psi) / np.sqrt(N)
    return Phi


def rho_sys(N: int, psi: np.ndarray) -> np.ndarray:
    """rho_sys(s) = (1/N) sum_g U_g psi psi^dag U_g^dag  (partial trace of |Phi><Phi|)."""
    acc = np.zeros((N, N), dtype=complex)
    for g in range(N):
        u = shift(N, g) @ psi
        acc += np.outer(u, u.conj())
    return acc / N


def random_state(d, rng):
    v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    return v / np.linalg.norm(v)


def random_hermitian(d, rng):
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    return (A + A.conj().T) / 2.0


def observables_R(F, k, kp):
    """Hermitian frame operators R_Re, R_Im on the N-dim frame register."""
    ek = F[:, k]
    ekp = F[:, kp]
    outer = np.outer(ek, ekp.conj())                    # |e_k><e_k'|
    R_Re = (outer + outer.conj().T) / 2.0
    R_Im = (outer - outer.conj().T) / (2j)
    return R_Re, R_Im


# ---------------------------------------------------------------------------
# per-case certificate
# ---------------------------------------------------------------------------
def certify_case(N, psi, M, F, k, kp, shots, rng):
    V = frame_components(N, psi, F)
    coh = complex(V[k].conj() @ M @ V[kp])              # analytic v_k^dag M v_{k'}
    Phi = joint_state(N, psi)
    R_Re, R_Im = observables_R(F, k, kp)
    Mnorm = float(np.linalg.norm(M, 2))                 # operator norm ||M||

    # resolution of identity: sum_k v_k v_k^dag == rho_sys(s)
    recon = sum(np.outer(V[j], V[j].conj()) for j in range(N))
    roi_err = float(np.max(np.abs(recon - rho_sys(N, psi))))

    mu, Meig = np.linalg.eigh(M)                         # system eigenbasis (single setting)
    out = dict(coh_re=float(coh.real), coh_im=float(coh.imag),
               Mnorm=Mnorm, roi_err=roi_err)

    for tag, R in (("re", R_Re), ("im", R_Im)):
        # (a) exact expectation <Phi| R (x) M |Phi>
        O = np.kron(R, M)
        exact = float(np.real(Phi.conj() @ O @ Phi))
        out[f"exact_{tag}"] = exact

        # (b) frame spectrum of R: must be {-1/2, 0, +1/2} with a 0-eigenspace of
        # dimension N-2 (single fixed measurement basis; zeros recorded, not dropped).
        r, Reig = np.linalg.eigh(R)
        r_sorted = np.sort(r)
        spec_err = max(abs(r_sorted[0] - (-0.5)),        # smallest eigenvalue = -1/2
                       abs(r_sorted[-1] - 0.5),          # largest  eigenvalue = +1/2
                       float(np.max(np.abs(r_sorted[1:-1]))))  # middle N-2 all = 0
        n_zero = int(np.sum(np.abs(r) < 1e-9))
        out[f"spec_err_{tag}"] = spec_err
        out[f"n_zero_eig_{tag}"] = n_zero

        # product-measurement outcome distribution (frame in R-basis, system in M-basis)
        Phi_mat = Phi.reshape(N, N)                      # [g_frame, i_system]
        Amp = Reig.conj().T @ Phi_mat @ Meig.conj()      # (frame_out, sys_out)
        probs = np.abs(Amp) ** 2
        prob_total = float(probs.sum())                  # == 1 exactly: no post-selection
        eigval_grid = np.outer(r, mu)                    # value r_a * mu_b
        flat_p = probs.ravel()
        flat_v = eigval_grid.ravel()

        # exact single-shot moments straight from the distribution
        m1 = float(flat_p @ flat_v)
        m2 = float(flat_p @ (flat_v ** 2))
        var_shot = m2 - m1 ** 2
        zmax = float(np.max(np.abs(flat_v)))             # single-shot |Z| range
        out[f"prob_total_{tag}"] = prob_total
        out[f"shotmean_{tag}"] = m1
        out[f"var_shot_{tag}"] = var_shot
        out[f"zmax_{tag}"] = zmax

        # (c) actual finite-shot Monte-Carlo (unbiasedness at n = O(1/eps^2))
        idx = rng.choice(flat_p.size, size=shots, p=flat_p / prob_total)
        samp = flat_v[idx]
        out[f"emp_mean_{tag}"] = float(samp.mean())
        out[f"emp_abserr_{tag}"] = abs(float(samp.mean()) - exact)

    # |coherence|^2 recovered from the two single-setting estimators
    out["abscoh2_analytic"] = out["coh_re"] ** 2 + out["coh_im"] ** 2
    out["abscoh2_from_exact"] = out["exact_re"] ** 2 + out["exact_im"] ** 2
    return out


# ---------------------------------------------------------------------------
# driver / certificate
# ---------------------------------------------------------------------------
def run(sizes, n_obs, shots, seeds, out_path):
    rows = []
    W = dict(exact=0.0, shotmean=0.0, spec=0.0, roi=0.0, prob=0.0,
             var_over_M2=0.0, var_over_M2q=0.0, abscoh2=0.0, emp=0.0, zbound=0.0)
    for N in sizes:
        F = fourier_frame(N)
        pairs = [(1, 2), (0, 1), (1, N - 1)]            # distinct cross-irrep pairs, k!=k'
        for sd in seeds:
            rng = np.random.default_rng(4000 + sd + 100000 * N)
            for _ in range(n_obs):
                phi = random_state(N, rng)
                s = int(rng.integers(0, N))
                psi = shift(N, s) @ phi                  # concept = U_s phi
                M = random_hermitian(N, rng)
                for (k, kp) in pairs:
                    d = certify_case(N, psi, M, F, k, kp, shots, rng)
                    Mn = d["Mnorm"]
                    W["roi"] = max(W["roi"], d["roi_err"])
                    abscoh2_err = abs(d["abscoh2_from_exact"] - d["abscoh2_analytic"])
                    W["abscoh2"] = max(W["abscoh2"], abscoh2_err)
                    for tag in ("re", "im"):
                        exact = d[f"exact_{tag}"]
                        coh = d["coh_re"] if tag == "re" else d["coh_im"]
                        exact_err = abs(exact - coh)              # (a) numeric==analytic
                        shotmean_err = abs(d[f"shotmean_{tag}"] - exact)
                        spec_err = d[f"spec_err_{tag}"]
                        prob_err = abs(d[f"prob_total_{tag}"] - 1.0)
                        var_shot = d[f"var_shot_{tag}"]
                        zmax = d[f"zmax_{tag}"]
                        # (c) bounds: Var <= ||M||^2/4 and |Z| <= ||M||/2
                        var_over_M2 = var_shot / (Mn ** 2 + 1e-300)
                        z_over_Mhalf = zmax / (0.5 * Mn + 1e-300)
                        emp_err = d[f"emp_abserr_{tag}"]
                        W["exact"] = max(W["exact"], exact_err)
                        W["shotmean"] = max(W["shotmean"], shotmean_err)
                        W["spec"] = max(W["spec"], spec_err)
                        W["prob"] = max(W["prob"], prob_err)
                        W["var_over_M2q"] = max(W["var_over_M2q"], var_over_M2)  # vs 1/4
                        W["zbound"] = max(W["zbound"], z_over_Mhalf)             # vs 1
                        W["emp"] = max(W["emp"], emp_err)
                        rows.append(dict(
                            group=f"Z{N}", N=N, k=k, kp=kp, part=tag, seed=sd,
                            analytic_coh=coh, exact_expect=exact,
                            exact_vs_analytic_err=exact_err,
                            shotmean_vs_exact_err=shotmean_err,
                            R_spec_err=spec_err, n_zero_eig=d[f"n_zero_eig_{tag}"],
                            prob_total=d[f"prob_total_{tag}"], prob_err=prob_err,
                            Mnorm=Mn, zmax=zmax, z_over_half_Mnorm=z_over_Mhalf,
                            var_single_shot=var_shot, var_over_Mnorm2=var_over_M2,
                            bound_quarter_ok=int(var_shot <= 0.25 * Mn ** 2 + 1e-12),
                            roi_err=d["roi_err"],
                            abscoh2_analytic=d["abscoh2_analytic"],
                            abscoh2_from_exact=d["abscoh2_from_exact"],
                            abscoh2_err=abscoh2_err,
                            shots=shots, emp_mean=d[f"emp_mean_{tag}"], emp_abs_err=emp_err,
                        ))
        sub = [r for r in rows if r["N"] == N]
        print(f"  Z{N}: max|exact-analytic|={max(r['exact_vs_analytic_err'] for r in sub):.2e}"
              f"  max R-spec-err={max(r['R_spec_err'] for r in sub):.2e}"
              f"  max Var/||M||^2={max(r['var_over_Mnorm2'] for r in sub):.3f}"
              f"  max|emp-exact|@{shots}={max(r['emp_abs_err'] for r in sub):.2e}"
              f"  ({len(sub)} rows)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tol = 1e-12
    print("\nwrote", out_path)
    print("=" * 72)
    print("CERTIFICATE SUMMARY  (LEG LA — single-setting post-selection-free estimator)")
    print("=" * 72)
    print(f"  (a) exactness         max|<Phi|O|Phi> - analytic Re/Im(v_k^H M v_k')| = "
          f"{W['exact']:.2e}   [tol {tol:.0e} -> {'PASS' if W['exact'] < tol else 'FAIL'}]")
    print(f"      consistency       max|distribution-mean - exact expect|          = "
          f"{W['shotmean']:.2e}   [tol {tol:.0e} -> {'PASS' if W['shotmean'] < tol else 'FAIL'}]")
    print(f"      resolution-of-id  max|sum_k v_k v_k^H - rho_sys|                  = "
          f"{W['roi']:.2e}   [tol {tol:.0e} -> {'PASS' if W['roi'] < tol else 'FAIL'}]")
    print(f"  (b) frame spectrum    max|spec(R) - {{-1/2,0,+1/2}}|                    = "
          f"{W['spec']:.2e}   [tol {tol:.0e} -> {'PASS' if W['spec'] < tol else 'FAIL'}]")
    print(f"      no post-selection max|sum of outcome probs - 1|                   = "
          f"{W['prob']:.2e}   [tol {tol:.0e} -> {'PASS' if W['prob'] < tol else 'FAIL'}]")
    print(f"  (c) single-shot range max |Z| / (||M||/2)                            = "
          f"{W['zbound']:.4f}   [bound <=1 -> {'PASS' if W['zbound'] <= 1 + 1e-9 else 'FAIL'}]")
    print(f"      variance bound    max Var_single_shot / ||M||^2                  = "
          f"{W['var_over_M2q']:.4f}   [bound <=1/4 -> {'PASS' if W['var_over_M2q'] <= 0.25 + 1e-9 else 'FAIL'}]")
    print(f"      |coh|^2 recovery  max|Re^2+Im^2 (exact) - |v_k^H M v_k'|^2|       = "
          f"{W['abscoh2']:.2e}   [tol {tol:.0e} -> {'PASS' if W['abscoh2'] < tol else 'FAIL'}]")
    print(f"      unbiasedness      max|emp mean - exact| @ {shots} shots (finite)  = "
          f"{W['emp']:.2e}   [Monte-Carlo, ~1/sqrt(n)]")
    # N-independence of the shot budget: bound n = ||M||^2/(4 eps^2) has no N.
    print(f"      N-independence    Var bound ||M||^2/4 has NO N-dependence; identical across "
          f"Z{sizes[0]}..Z{sizes[-1]} (max Var/||M||^2={W['var_over_M2q']:.4f} for all N)")
    return rows, W


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 7, 8, 16])
    ap.add_argument("--n_obs", type=int, default=6)
    ap.add_argument("--shots", type=int, default=200000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/fisher_information/certificates/la_cert.csv")
    args = ap.parse_args(argv)
    run(args.sizes, args.n_obs, args.shots, args.seeds, args.out)


if __name__ == "__main__":
    main()
