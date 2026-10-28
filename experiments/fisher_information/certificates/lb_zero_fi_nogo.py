"""
LEG LB -- Zero-Fisher-Information no-go for the decohered / classical-FA marginal.
=================================================================================
Claim (unconditional): for G = Z_N acting by the shift (regular) representation,
the system marginal of the frame+system state

    |Phi_s> = |G|^{-1/2} sum_g |g>_R (x) U_g psi_s ,   psi_s = U_s phi ,

    rho_sys(s) = Tr_R |Phi_s><Phi_s| = (1/N) sum_g U_g psi_s psi_s^dag U_g^dag
               = (1/N) sum_g U_g U_s phi phi^dag U_s^dag U_g^dag

is EXACTLY independent of the hidden shift s.  Hence every classical/quantum
Fisher information about s carried by rho_sys is identically 0: the decohered
marginal (what classical frame averaging / Mode B sees) is information-theoretically
blind to the hidden shift.

This script produces the numeric certificate:
  (a) DISCRETE  : ||rho_sys(s) - rho_sys(0)||  < 1e-12  for all s in Z_N,
                  all groups {Z4, Z8, Z16}, many random phi.  (group-closure core)
  (b) numeric==analytic : rho_sys(s) equals the Fourier-diagonal twirl of rho0
                  to ~1e-12 (closed form).
  (c) CONTINUOUS: embed a U(1) shift subfamily U(theta)=exp(i theta A), A=-i log P,
                  and certify  d rho_sys/d theta = 0  and  QFI(rho_sys,theta)=0
                  to 1e-10.
  (d) classical FI = 0 for an explicit informationally-complete-ish measurement,
                  because the outcome law p(x|s) is constant in s.

Every reported number is written to results/fisher_information/certificates/lb_zero_fi_nogo.csv .
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


def rand_state(N: int, rng) -> np.ndarray:
    v = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    return v / np.linalg.norm(v)


def rho_sys(phi: np.ndarray, N: int, s: int) -> np.ndarray:
    """rho_sys(s) = (1/N) sum_g U_g U_s |phi><phi| U_s^dag U_g^dag."""
    psi = shift(N, s) @ phi
    P0 = np.outer(psi, psi.conj())
    acc = np.zeros((N, N), dtype=complex)
    for g in range(N):
        Ug = shift(N, g)
        acc += Ug @ P0 @ Ug.conj().T
    return acc / N


def twirl_analytic(phi: np.ndarray, N: int) -> np.ndarray:
    """Closed form: G-twirl keeps only the Fourier-diagonal of rho0.
    In the DFT basis F (F[k,x]=w^{kx}/sqrt(N), w=exp(-2pi i/N)) the twirl of any X
    is diag_Fourier(X): (1/N) sum_g U_g X U_g^dag = F^dag diag(F X F^dag)_kk F ... .
    We build it directly by projecting onto Fourier-diagonal."""
    w = np.exp(-2j * np.pi / N)
    F = np.array([[w ** (k * x) for x in range(N)] for k in range(N)]) / np.sqrt(N)
    rho0 = np.outer(phi, phi.conj())
    Xf = F @ rho0 @ F.conj().T          # rho0 in Fourier basis
    Xf_diag = np.diag(np.diag(Xf))      # keep only k=k'
    return F.conj().T @ Xf_diag @ F     # back to computational basis


def continuous_shift_generator(N: int) -> np.ndarray:
    """A Hermitian A with exp(i A) = P (the unit cyclic shift), via spectral log.
    U(theta) = exp(i theta A) is a U(1) subfamily with U(1)=P, U(0)=I."""
    P = shift(N, 1)
    evals, V = np.linalg.eig(P)               # P is unitary, normal
    ang = np.angle(evals)                      # exp(i ang) = evals
    A = (V @ np.diag(ang) @ np.linalg.inv(V))
    A = (A + A.conj().T) / 2.0                 # Hermitize (numerical)
    return A


def rho_sys_theta(phi: np.ndarray, N: int, theta: float, A: np.ndarray) -> np.ndarray:
    """rho_sys(theta) = (1/N) sum_g U_g U(theta) rho0 U(theta)^dag U_g^dag."""
    U = _expm_herm(theta, A)
    psi = U @ phi
    P0 = np.outer(psi, psi.conj())
    acc = np.zeros((N, N), dtype=complex)
    for g in range(N):
        Ug = shift(N, g)
        acc += Ug @ P0 @ Ug.conj().T
    return acc / N


def _expm_herm(theta: float, A: np.ndarray) -> np.ndarray:
    ev, V = np.linalg.eigh(A)
    return V @ np.diag(np.exp(1j * theta * ev)) @ V.conj().T


def qfi_theta(phi: np.ndarray, N: int, A: np.ndarray, theta: float = 0.3,
              h: float = 1e-4) -> tuple[float, float]:
    """Quantum Fisher information of rho_sys(theta) about theta at `theta`.
    QFI = 2 sum_{ij} |<i|dRho|j>|^2 / (lam_i+lam_j)  (sum over lam_i+lam_j>0).
    Returns (QFI, ||dRho||_F) with dRho a central finite difference."""
    rp = rho_sys_theta(phi, N, theta + h, A)
    rm = rho_sys_theta(phi, N, theta - h, A)
    dR = (rp - rm) / (2 * h)
    r0 = rho_sys_theta(phi, N, theta, A)
    lam, U = np.linalg.eigh((r0 + r0.conj().T) / 2)
    dR_e = U.conj().T @ dR @ U
    qfi = 0.0
    for i in range(N):
        for j in range(N):
            denom = lam[i] + lam[j]
            if denom > 1e-12:
                qfi += 2.0 * (abs(dR_e[i, j]) ** 2) / denom
    return float(qfi), float(np.linalg.norm(dR))


def classical_fi(phi: np.ndarray, N: int, s_grid) -> float:
    """Max over a family of projective measurements of the classical FI about s,
    using the constancy of p(x|s). We measure rho_sys(s) in the computational and
    Fourier bases and in a random basis; FI = sum_x (dp/ds)^2/p ~ 0 since p const.
    Returned: max finite-difference classical FI over the three bases."""
    rng = np.random.default_rng(12345)
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    w = np.exp(-2j * np.pi / N)
    F = np.array([[w ** (k * x) for x in range(N)] for k in range(N)]) / np.sqrt(N)
    bases = {"comp": np.eye(N, dtype=complex), "fourier": F, "random": Q.conj().T}
    max_fi = 0.0
    for _, B in bases.items():
        # p(x|s) = <x_B| rho_sys(s) |x_B> = diag(B rho B^dag)
        ps = []
        for s in s_grid:
            r = rho_sys(phi, N, s)
            ps.append(np.real(np.diag(B @ r @ B.conj().T)))
        ps = np.array(ps)  # (len(s_grid), N)
        # finite-difference FI between consecutive discrete s (a proxy; p is const)
        for a in range(len(s_grid) - 1):
            dp = ps[a + 1] - ps[a]
            p = np.clip(ps[a], 1e-15, None)
            max_fi = max(max_fi, float(np.sum(dp * dp / p)))
    return max_fi


def main():
    groups = [4, 8, 16]
    n_phi = 20
    rows = []
    print("LEG LB -- Zero-FI no-go certificate")
    print("=" * 60)

    global_max_discrete = 0.0
    global_max_numvsana = 0.0
    global_max_qfi = 0.0
    global_max_ddtheta = 0.0
    global_max_cfi = 0.0
    global_max_tracedist = 0.0

    for N in groups:
        A = continuous_shift_generator(N)
        rng = np.random.default_rng(1000 + N)
        max_discrete = 0.0
        max_numvsana = 0.0
        max_qfi = 0.0
        max_ddtheta = 0.0
        max_cfi = 0.0
        max_tracedist = 0.0
        for _ in range(n_phi):
            phi = rand_state(N, rng)
            r0 = rho_sys(phi, N, 0)
            ana = twirl_analytic(phi, N)
            max_numvsana = max(max_numvsana, float(np.linalg.norm(r0 - ana)))
            for s in range(N):
                rs = rho_sys(phi, N, s)
                diff = float(np.linalg.norm(rs - r0))
                max_discrete = max(max_discrete, diff)
                # exact trace distance D(rho(s),rho(0)) = 0.5 * ||.||_1
                ev = np.linalg.eigvalsh((rs - r0 + (rs - r0).conj().T) / 2)
                max_tracedist = max(max_tracedist, 0.5 * float(np.sum(np.abs(ev))))
            q, dd = qfi_theta(phi, N, A)
            max_qfi = max(max_qfi, q)
            max_ddtheta = max(max_ddtheta, dd)
            max_cfi = max(max_cfi, classical_fi(phi, N, list(range(N))))

        rows.append(dict(
            group=f"Z{N}", N=N, n_phi=n_phi,
            max_discrete_frob=max_discrete,
            max_trace_distance=max_tracedist,
            max_num_vs_analytic=max_numvsana,
            max_qfi_theta=max_qfi,
            max_ddtheta_frob=max_ddtheta,
            max_classical_fi=max_cfi,
        ))
        global_max_discrete = max(global_max_discrete, max_discrete)
        global_max_numvsana = max(global_max_numvsana, max_numvsana)
        global_max_qfi = max(global_max_qfi, max_qfi)
        global_max_ddtheta = max(global_max_ddtheta, max_ddtheta)
        global_max_cfi = max(global_max_cfi, max_cfi)
        global_max_tracedist = max(global_max_tracedist, max_tracedist)
        print(f"Z{N:<3} | ||rho(s)-rho(0)||<= {max_discrete:.2e} | trace-dist<= "
              f"{max_tracedist:.2e} | num-vs-analytic {max_numvsana:.2e} | "
              f"QFI(theta)<= {max_qfi:.2e} | ||dRho/dtheta||<= {max_ddtheta:.2e} | "
              f"classFI<= {max_cfi:.2e}")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "lb_zero_fi_nogo.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("=" * 60)
    print(f"GLOBAL: discrete<= {global_max_discrete:.2e}  trace-dist<= {global_max_tracedist:.2e}  "
          f"num-vs-analytic<= {global_max_numvsana:.2e}")
    print(f"        QFI<= {global_max_qfi:.2e}  ||dRho/dtheta||<= {global_max_ddtheta:.2e}  "
          f"classFI<= {global_max_cfi:.2e}")
    thresholds = {
        "discrete<1e-12": global_max_discrete < 1e-12,
        "trace_dist<1e-12": global_max_tracedist < 1e-12,
        "num_vs_analytic<1e-12": global_max_numvsana < 1e-12,
        "QFI<1e-10": global_max_qfi < 1e-10,
        "ddtheta<1e-10": global_max_ddtheta < 1e-10,
        "classFI<1e-10": global_max_cfi < 1e-10,
    }
    print("CERTIFICATE:", "PASS" if all(thresholds.values()) else "FAIL", thresholds)
    print("wrote", path)


if __name__ == "__main__":
    main()
