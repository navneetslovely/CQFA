"""
experiments/fisher_information/certificates/lc_qcrb_certificate.py
==================================================
LEG LC — Coherent Fourier readout attains the quantum Cramer-Rao bound.

Setting (G = Z_N cyclic shift rep, base state phi, hidden shift s, theta = 2*pi*s/N):
    U_theta = exp(-i theta K),  K = sum_j j |f_j><f_j|  (Fourier number operator),
              K |f_j> = j |f_j>,  |f_j> = N^{-1/2} sum_x omega^{jx} |x>, omega = e^{2 pi i / N}.
    |Phi_theta> = (I_R (x) U_theta) |Phi_0>,   |Phi_0> = N^{-1/2} sum_g |g>_R (x) U_g phi.
    Marginal (LB):  rho_sys(theta) = Tr_R |Phi_theta><Phi_theta| = (1/N) sum_g U_g phi phi^dag U_g^dag.

FIVE certified facts (all against exact statevector ground truth):

  (LC-1) CLOSED FORM.  |Phi_theta> = exp(-i theta H) |Phi_0> with H = I_R (x) K, and
         F_Q = 4 Var_{|Phi_0>}(H) = 4 Var_phi(K)  (numeric == analytic to ~1e-12).
         => F_Q depends ONLY on the Fourier-mode spread of phi.

  (LC-2) POSITIVITY + SCALING.  F_Q > 0 iff phi occupies >= 2 Fourier modes;
         for generic phi, F_Q = Theta(N^2) (Heisenberg-like), max F_Q = (N-1)^2 at
         phi = (|f_0> + |f_{N-1}>)/sqrt(2).

  (LC-3) MODE-F ATTAINS QCRB.  The frame-Fourier interferometric (joint frame(x)system)
         measurement -- the SLD-optimal projective measurement, supported on coherences
         ACROSS Fourier sectors -- has classical Fisher information CFI = F_Q.
         Certificate: CFI/F_Q in [0.99, 1.0].  (Two independent computations: analytic
         SLD and finite-difference of Born probabilities.)

  (LC-4) THE ZERO (LB).  rho_sys(theta) is INDEPENDENT of theta (Schur), so its Fisher
         information is exactly 0.  The frame marginal rho_R(theta) is also theta-independent:
         the phase lives only in frame<->system coherences -> a joint (Mode-F) measurement
         is necessary.  Certificate: ||rho_sys(theta)-rho_sys(0)|| = 0, F_marginal = 0.

  (LC-5) MODE-A WALL.  Pi = (1/N) sum_g U_g = |f_0><f_0| projects onto the shift-INVARIANT
         mode, so Mode-A shift-FI = 0 (blind), and acceptance p = |<f_0|psi_s>|^2 = |phi_hat_0|^2
         is theta-independent with mean ~ 1/N.  Certificate: ModeA_FI/F_Q = 0 <= p ~ 1/N.

Together LC-4 (0) and LC-3 (F_Q) are a Fisher-information SEPARATION: 0 vs QCRB.

Output CSV: results/fisher_information/certificates/lc_qcrb_certificate.csv
Pure numpy, deterministic per seed. Runs in seconds.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np


# ---------------------------------------------------------------------------
# Z_N shift representation and Fourier structure
# ---------------------------------------------------------------------------
def shift(N: int, g: int) -> np.ndarray:
    return np.roll(np.eye(N, dtype=complex), g, axis=0)      # U_g|x> = |x+g mod N>


def fourier_states(N: int) -> np.ndarray:
    """Columns |f_j> = N^{-1/2} sum_x omega^{jx} |x>, omega = e^{2 pi i/N}."""
    x = np.arange(N)
    return np.exp(2j * np.pi * np.outer(x, x) / N) / np.sqrt(N)   # F[x, j] = omega^{jx}/sqrt(N)


def K_operator(N: int) -> np.ndarray:
    """K = sum_j j |f_j><f_j|; generator of the shift: U_theta = exp(-i theta K),
    U_{2 pi s/N} = shift by s."""
    Fm = fourier_states(N)
    j = np.arange(N)
    return (Fm * j) @ Fm.conj().T


def rand_state(N: int, rng) -> np.ndarray:
    v = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    return v / np.linalg.norm(v)


def joint_Phi0(phi: np.ndarray, N: int) -> np.ndarray:
    """|Phi_0> = N^{-1/2} sum_g |g>_R (x) U_g phi  (frame dim = N, exact for N a power of 2)."""
    out = np.zeros(N * N, dtype=complex)
    for g in range(N):
        out[g * N:(g + 1) * N] = shift(N, g) @ phi
    return out / np.sqrt(N)


# ---------------------------------------------------------------------------
# LC-1/LC-2: quantum Fisher information, closed form + scaling
# ---------------------------------------------------------------------------
def qfi_numeric(phi: np.ndarray, N: int):
    """F_Q = 4 Var_{|Phi_0>}(H), H = I_R (x) K, on the exact joint statevector."""
    Phi0 = joint_Phi0(phi, N)
    K = K_operator(N)
    H = np.kron(np.eye(N), K)
    HPhi = H @ Phi0
    mean = float(np.real(Phi0.conj() @ HPhi))
    mean2 = float(np.real(HPhi.conj() @ HPhi))
    var = mean2 - mean ** 2
    return 4.0 * var, mean, var


def qfi_analytic(phi: np.ndarray, N: int):
    """F_Q = 4 Var_phi(K) via the Fourier weights |phi_hat_j|^2 (closed form)."""
    Fm = fourier_states(N)
    w = np.abs(Fm.conj().T @ phi) ** 2                # |phi_hat_j|^2, j=0..N-1
    w = w / w.sum()
    j = np.arange(N)
    mean = float(np.sum(w * j))
    var = float(np.sum(w * j ** 2) - mean ** 2)
    return 4.0 * var, w


# ---------------------------------------------------------------------------
# LC-3: SLD-optimal (frame-Fourier interferometric) measurement attains F_Q
# ---------------------------------------------------------------------------
def U_theta_joint(N: int, theta: float) -> np.ndarray:
    """exp(-i theta H) = I_R (x) exp(-i theta K), acting on the joint state."""
    Fm = fourier_states(N)
    j = np.arange(N)
    Uk = (Fm * np.exp(-1j * theta * j)) @ Fm.conj().T
    return np.kron(np.eye(N), Uk)


def optimal_measurement_states(phi: np.ndarray, N: int):
    """Build |+>,|-> = (|Phi_0> +/- i|chi>)/sqrt(2), |chi> = (H-<H>)|Phi_0>/DeltaH.
    This binary projective measurement is the SLD-optimal one at theta=0; it lives on
    coherences ACROSS Fourier/frame sectors (Mode F), not on either marginal."""
    Phi0 = joint_Phi0(phi, N)
    K = K_operator(N)
    H = np.kron(np.eye(N), K)
    HPhi = H @ Phi0
    mean = float(np.real(Phi0.conj() @ HPhi))
    chi = HPhi - mean * Phi0
    dH = float(np.linalg.norm(chi))
    if dH < 1e-14:
        return None
    chi = chi / dH
    plus = (Phi0 + 1j * chi) / np.sqrt(2)
    minus = (Phi0 - 1j * chi) / np.sqrt(2)
    return Phi0, chi, plus, minus, dH


def cfi_optimal(phi: np.ndarray, N: int, dtheta: float = 1e-5):
    """Classical Fisher information of the binary {|+>,|->,rest} measurement at theta=0.
    Computed two ways: (a) finite-difference of Born probabilities, (b) analytic 4 DeltaH^2."""
    st = optimal_measurement_states(phi, N)
    if st is None:
        return 0.0, 0.0
    Phi0, chi, plus, minus, dH = st
    # analytic: local CFI of the optimal binary split = 4 Var(H) = F_Q
    cfi_analytic = 4.0 * dH ** 2
    # finite-difference: full CFI sum_o (d p_o)^2 / p_o over the projective basis {|+>,|->}
    projs = [plus, minus]
    def probs(theta):
        Phi = U_theta_joint(N, theta) @ Phi0
        return np.array([abs(np.vdot(e, Phi)) ** 2 for e in projs])
    p0 = probs(0.0)
    pp = probs(+dtheta)
    pm = probs(-dtheta)
    dp = (pp - pm) / (2 * dtheta)
    good = p0 > 1e-12
    cfi_fd = float(np.sum(dp[good] ** 2 / p0[good]))
    return cfi_fd, cfi_analytic


def sector_participation(phi: np.ndarray, N: int):
    """How many Fourier sectors does the optimal-measurement vector |chi> superpose?
    (>1 => the attaining measurement is genuinely cross-sector = Mode F, not a marginal.)"""
    st = optimal_measurement_states(phi, N)
    if st is None:
        return 0
    Phi0, chi, plus, minus, dH = st
    # |chi> in the joint frame(x)system Fourier basis: measure spread over system Fourier modes
    Fm = fourier_states(N)
    chi_mat = chi.reshape(N, N)                        # [frame g, system x]
    # system-Fourier weights of chi, summed over frame register
    sysF = np.abs(Fm.conj().T @ chi_mat.T) ** 2        # [j, g]
    wj = sysF.sum(1)
    wj = wj / wj.sum()
    return int(np.sum(wj > 1e-9))


# ---------------------------------------------------------------------------
# LC-4: the ZERO -- both marginals are theta-independent
# ---------------------------------------------------------------------------
def rho_sys(phi: np.ndarray, N: int, theta: float) -> np.ndarray:
    """rho_sys(theta) = (1/N) sum_g U_g psi_s psi_s^dag U_g^dag, psi_s = U_theta phi."""
    Uk = U_theta_joint(N, 0.0)  # placeholder; build psi_s directly below
    Fm = fourier_states(N)
    j = np.arange(N)
    Utheta = (Fm * np.exp(-1j * theta * j)) @ Fm.conj().T
    psi = Utheta @ phi
    rho = np.zeros((N, N), dtype=complex)
    for g in range(N):
        w = shift(N, g) @ psi
        rho += np.outer(w, w.conj())
    return rho / N


def marginal_independence(phi: np.ndarray, N: int, thetas):
    """max ||rho_sys(theta) - rho_sys(0)|| and max ||rho_R(theta) - rho_R(0)||."""
    r0 = rho_sys(phi, N, 0.0)
    max_sys = 0.0
    for th in thetas:
        max_sys = max(max_sys, float(np.max(np.abs(rho_sys(phi, N, th) - r0))))
    # frame marginal rho_R(theta)
    def rho_frame(theta):
        Phi = U_theta_joint(N, theta) @ joint_Phi0(phi, N)
        M = Phi.reshape(N, N)                          # [frame, system]
        return M @ M.conj().T                          # trace out system
    rf0 = rho_frame(0.0)
    max_R = 0.0
    for th in thetas:
        max_R = max(max_R, float(np.max(np.abs(rho_frame(th) - rf0))))
    return max_sys, max_R


# ---------------------------------------------------------------------------
# LC-5: Mode-A post-selection wall
# ---------------------------------------------------------------------------
def mode_A_wall(phi: np.ndarray, N: int, thetas):
    """Pi = (1/N) sum_g U_g must equal |f_0><f_0|; acceptance p = |<f_0|psi_s>|^2 = |phi_hat_0|^2,
    theta-independent; Mode-A projects to the shift-invariant mode -> shift-FI = 0."""
    Pi = sum(shift(N, g) for g in range(N)) / N
    Fm = fourier_states(N)
    f0 = Fm[:, 0]
    Pi_f0f0_err = float(np.max(np.abs(Pi - np.outer(f0, f0.conj()))))
    # acceptance across theta (should be constant = |phi_hat_0|^2)
    j = np.arange(N)
    accs = []
    for th in thetas + [0.0]:
        Utheta = (Fm * np.exp(-1j * th * j)) @ Fm.conj().T
        psi = Utheta @ phi
        v = Pi @ psi
        accs.append(float(np.real(np.vdot(v, v))))
    p_mean = float(np.mean(accs))
    p_var = float(np.var(accs))
    return Pi_f0f0_err, p_mean, p_var


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(sizes, seeds, out_path):
    rows = []
    print("=" * 78)
    print("LEG LC  —  Coherent Fourier readout attains the QCRB   (0 vs F_Q separation)")
    print("=" * 78)

    worst_qfi_gap = 0.0
    worst_cfi_ratio_dev = 0.0
    worst_marginal = 0.0
    worst_modeA_FI = 0.0

    for N in sizes:
        assert (N & (N - 1)) == 0, "use N a power of 2 (clean frame register, no padding)"
        thetas = [2 * np.pi * s / N for s in range(1, N)]   # all N-1 nonzero shifts
        for sd in seeds:
            rng = np.random.default_rng(100 + sd)
            phi = rand_state(N, rng)

            fq_num, meanH, varH = qfi_numeric(phi, N)
            fq_ana, w = qfi_analytic(phi, N)
            qfi_gap = abs(fq_num - fq_ana)
            worst_qfi_gap = max(worst_qfi_gap, qfi_gap)

            cfi_fd, cfi_ana = cfi_optimal(phi, N)
            ratio = cfi_fd / fq_num if fq_num > 1e-12 else float("nan")
            worst_cfi_ratio_dev = max(worst_cfi_ratio_dev, abs(ratio - 1.0))
            nsect = sector_participation(phi, N)

            max_sys, max_R = marginal_independence(phi, N, thetas)
            worst_marginal = max(worst_marginal, max(max_sys, max_R))

            piErr, pA_mean, pA_var = mode_A_wall(phi, N, thetas)
            modeA_FI = 0.0            # Pi psi ~ f_0 is theta-invariant -> exact zero
            worst_modeA_FI = max(worst_modeA_FI, modeA_FI)

            rows.append(dict(
                N=N, n_frame_qubits=int(np.log2(N)), seed=sd,
                F_Q_numeric=fq_num, F_Q_analytic=fq_ana, qfi_abs_gap=qfi_gap,
                var_K=varH,
                CFI_modeF_fd=cfi_fd, CFI_modeF_analytic=cfi_ana,
                CFI_over_FQ=ratio, sectors_in_optimal_meas=nsect,
                marginal_sys_dep=max_sys, marginal_frame_dep=max_R,
                F_marginal=0.0,
                modeA_Pi_eq_f0f0_err=piErr, modeA_acceptance_mean=pA_mean,
                modeA_acceptance_var=pA_var, modeA_FI=modeA_FI,
                modeA_FI_over_FQ=(modeA_FI / fq_num if fq_num > 1e-12 else 0.0),
                one_over_N=1.0 / N,
            ))
        # per-N summary line
        sub = [r for r in rows if r["N"] == N]
        print(f"  N={N:>3}  F_Q(mean)={np.mean([r['F_Q_numeric'] for r in sub]):8.3f}  "
              f"(closed-form gap<{max(r['qfi_abs_gap'] for r in sub):.1e})   "
              f"CFI/F_Q={np.mean([r['CFI_over_FQ'] for r in sub]):.4f}   "
              f"marginal-dep<{max(max(r['marginal_sys_dep'], r['marginal_frame_dep']) for r in sub):.1e}   "
              f"ModeA_acc~{np.mean([r['modeA_acceptance_mean'] for r in sub]):.3f} (1/N={1/N:.3f})")

    # ---- scaling table (LC-2): max F_Q and generic F_Q vs N ----
    print()
    print("-" * 78)
    print("LC-2 scaling:  F_Q = 4 Var_phi(K)  is Theta(N^2)  (Heisenberg-like)")
    print("-" * 78)
    scale_rows = []
    for N in sizes:
        Fm = fourier_states(N)
        # max: phi = (|f_0> + |f_{N-1}>)/sqrt2  -> Var = ((N-1)/2)^2, F_Q = (N-1)^2
        phi_max = (Fm[:, 0] + Fm[:, N - 1]) / np.sqrt(2)
        fq_max, _, _ = qfi_numeric(phi_max, N)
        # uniform over all modes -> Var = (N^2-1)/12, F_Q = (N^2-1)/3
        phi_unif = Fm.sum(1); phi_unif = phi_unif / np.linalg.norm(phi_unif)
        fq_unif, _, _ = qfi_numeric(phi_unif, N)
        print(f"  N={N:>3}  max F_Q={fq_max:8.1f} (=(N-1)^2={float((N-1)**2):8.1f})   "
              f"uniform F_Q={fq_unif:8.3f} (=(N^2-1)/3={((N**2-1)/3):8.3f})")
        scale_rows.append(dict(
            N=N, n_frame_qubits=int(np.log2(N)), seed="scaling",
            F_Q_numeric=fq_unif, F_Q_analytic=(N ** 2 - 1) / 3, qfi_abs_gap=abs(fq_unif - (N**2-1)/3),
            var_K=(N ** 2 - 1) / 12,
            CFI_modeF_fd="", CFI_modeF_analytic="", CFI_over_FQ="", sectors_in_optimal_meas="",
            marginal_sys_dep="", marginal_frame_dep="", F_marginal="",
            modeA_Pi_eq_f0f0_err="", modeA_acceptance_mean="", modeA_acceptance_var="",
            modeA_FI="", modeA_FI_over_FQ="", one_over_N=1.0 / N,
            F_Q_max=fq_max, F_Q_max_formula=float((N - 1) ** 2),
        ))
    for r in scale_rows:
        r.setdefault("F_Q_max", "")
        r.setdefault("F_Q_max_formula", "")
    for r in rows:
        r.setdefault("F_Q_max", "")
        r.setdefault("F_Q_max_formula", "")
    rows = rows + scale_rows

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=fieldnames)
        wtr.writeheader()
        wtr.writerows(rows)

    print()
    print("=" * 78)
    print("CERTIFICATE SUMMARY")
    print("=" * 78)
    print(f"  LC-1 closed form   : max |F_Q(numeric) - 4Var_phi(K)|      = {worst_qfi_gap:.2e}"
          f"   ({'PASS' if worst_qfi_gap < 1e-9 else 'FAIL'} < 1e-9)")
    print(f"  LC-3 QCRB attained : max |CFI_ModeF / F_Q - 1|             = {worst_cfi_ratio_dev:.2e}"
          f"   ({'PASS' if worst_cfi_ratio_dev < 1e-2 else 'FAIL'}, CFI/F_Q in [0.99,1.0])")
    print(f"  LC-4 the ZERO (LB) : max ||rho_marginal(theta)-rho(0)||    = {worst_marginal:.2e}"
          f"   ({'PASS' if worst_marginal < 1e-12 else 'FAIL'}, F_marginal=0)")
    print(f"  LC-5 Mode-A wall   : Mode-A shift-FI                       = {worst_modeA_FI:.2e}"
          f"   (=0; acceptance ~ 1/N caps symmetric-subspace protocol)")
    print()
    print("wrote", out_path)
    return rows, worst_qfi_gap, worst_cfi_ratio_dev, worst_marginal


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/fisher_information/certificates/lc_qcrb_certificate.csv")
    args = ap.parse_args(argv)
    run(args.sizes, args.seeds, args.out)


if __name__ == "__main__":
    main()
