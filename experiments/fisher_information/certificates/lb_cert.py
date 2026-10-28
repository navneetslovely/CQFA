"""
LEG LB -- Zero-Fisher-Information no-go (unconditional centerpiece), certificate.
=================================================================================
Statement (unconditional).  Let G = Z_N act on C^N by the regular (shift)
representation U_g|x> = |x+g mod N>.  Fix phi with ||phi||=1; for a hidden shift
s set psi_s = U_s phi and form the frame+system state

    |Phi_s> = N^{-1/2} sum_{g in G} |g>_R (x) U_g psi_s .

Then the system marginal

    rho_sys(s) = Tr_R |Phi_s><Phi_s| = (1/N) sum_g U_g psi_s psi_s^dag U_g^dag
               = (1/N) sum_g U_{g+s} phi phi^dag U_{g+s}^dag

equals rho_sys(0) EXACTLY for every s and every phi (group-closure reindexing).
Consequences certified here numerically against closed forms:

  (LB-core) DISCRETE      : ||rho_sys(s) - rho_sys(0)||_F = 0            (<1e-12)
  (LB-i)    CLASSICAL FI   : p(x|s)=Tr(E_x rho_sys(s)) constant in s
                            => F_cl(s) = 0 for EVERY POVM               (<1e-12)
  (LB-ii)   TRACE DISTANCE : D(rho_sys(s),rho_sys(s')) = 0              (<1e-12)
  (LB-ana)  NUM==ANALYTIC  : rho_sys(s) equals the Fourier-diagonal G-twirl
                            T(rho_0) of rho_0 = phi phi^dag             (<1e-12)
  (LB-iii)  CONTINUOUS QFI : for U(theta)=exp(i theta A), A=-i Log(U_1), the
                            twirled marginal rho_sys(theta)=T(rho_0) is
                            theta-independent => d rho/dtheta = 0, QFI = 0
                            (numeric ~1e-10; central finite difference)

Groups: Z4, Z8, Z16 and a prime-order group Z7 (regular rep; the prime case is
the cyclic group underlying a DLP instance).  Every reported number is written to
results/fisher_information/certificates/lb_cert.csv .  ONLY numbers printed / in the CSV are
claimed; closed forms are checked to ~1e-12.
"""
from __future__ import annotations

import csv
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO, "results", "fisher_information", "certificates")


# ---------------------------------------------------------------------------
# regular representation of Z_N and helpers
# ---------------------------------------------------------------------------
def shift(N: int, g: int) -> np.ndarray:
    """U_g |x> = |x+g mod N>  (cyclic shift / regular rep of Z_N)."""
    return np.roll(np.eye(N, dtype=complex), g, axis=0)


def rand_state(N: int, rng) -> np.ndarray:
    v = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    return v / np.linalg.norm(v)


def rho_sys(phi: np.ndarray, N: int, s: int) -> np.ndarray:
    """rho_sys(s) = (1/N) sum_g U_g (U_s phi)(U_s phi)^dag U_g^dag."""
    psi = shift(N, s) @ phi
    P0 = np.outer(psi, psi.conj())
    acc = np.zeros((N, N), dtype=complex)
    for g in range(N):
        Ug = shift(N, g)
        acc += Ug @ P0 @ Ug.conj().T
    return acc / N


def dft(N: int) -> np.ndarray:
    """Unitary DFT F with F[k,x] = w^{kx}/sqrt(N), w = exp(-2 pi i / N).
    The U_g are simultaneously diagonalized: F U_g F^dag = diag(w^{-gk})."""
    w = np.exp(-2j * np.pi / N)
    return np.array([[w ** (k * x) for x in range(N)] for k in range(N)]) / np.sqrt(N)


def twirl_analytic(phi: np.ndarray, N: int) -> np.ndarray:
    """Closed form of the G-twirl T(rho_0): keep only the Fourier-diagonal of
    rho_0 = phi phi^dag.  T(X) = F^dag diag(diag(F X F^dag)) F ."""
    F = dft(N)
    rho0 = np.outer(phi, phi.conj())
    Xf = F @ rho0 @ F.conj().T
    return F.conj().T @ np.diag(np.diag(Xf)) @ F


# ---------------------------------------------------------------------------
# continuous U(1) shift subfamily: A = -i Log(U_1), U(theta)=exp(i theta A)
# ---------------------------------------------------------------------------
def shift_generator(N: int) -> np.ndarray:
    """Hermitian A with exp(i A) = U_1 (principal matrix log via spectral calc).
    A is a spectral function of U_1, hence commutes with every U_g."""
    P = shift(N, 1)
    evals, V = np.linalg.eig(P)          # unitary => normal, eigenbasis exists
    ang = np.angle(evals)                # principal branch: exp(i ang) = evals
    A = V @ np.diag(ang) @ np.linalg.inv(V)
    return (A + A.conj().T) / 2.0        # Hermitize away round-off


def expm_herm(theta: float, A: np.ndarray) -> np.ndarray:
    ev, V = np.linalg.eigh(A)
    return V @ np.diag(np.exp(1j * theta * ev)) @ V.conj().T


def rho_sys_theta(phi: np.ndarray, N: int, theta: float, A: np.ndarray) -> np.ndarray:
    U = expm_herm(theta, A)
    psi = U @ phi
    P0 = np.outer(psi, psi.conj())
    acc = np.zeros((N, N), dtype=complex)
    for g in range(N):
        Ug = shift(N, g)
        acc += Ug @ P0 @ Ug.conj().T
    return acc / N


def qfi_and_dtheta(phi: np.ndarray, N: int, A: np.ndarray,
                   theta: float = 0.37, h: float = 1e-4) -> tuple[float, float]:
    """QFI(rho_sys(theta), theta) and ||d rho/dtheta||_F via central difference.
    QFI = 2 sum_{ij} |<i|dRho|j>|^2 / (lam_i+lam_j)  over lam_i+lam_j > 0."""
    rp = rho_sys_theta(phi, N, theta + h, A)
    rm = rho_sys_theta(phi, N, theta - h, A)
    dR = (rp - rm) / (2 * h)
    r0 = rho_sys_theta(phi, N, theta, A)
    lam, U = np.linalg.eigh((r0 + r0.conj().T) / 2)
    dR_e = U.conj().T @ dR @ U
    qfi = 0.0
    for i in range(N):
        for j in range(N):
            d = lam[i] + lam[j]
            if d > 1e-12:
                qfi += 2.0 * (abs(dR_e[i, j]) ** 2) / d
    return float(qfi), float(np.linalg.norm(dR))


# ---------------------------------------------------------------------------
# classical Fisher information about s for arbitrary POVMs / bases
# ---------------------------------------------------------------------------
def classical_fi_max(phi: np.ndarray, N: int) -> float:
    """Max finite-difference classical FI about s over three measurement bases
    (computational, Fourier, random unitary).  Since p(x|s)=diag(B rho(s) B^dag)
    is constant in s, F_cl = sum_x (dp/ds)^2 / p = 0."""
    rng = np.random.default_rng(2718)
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    bases = {"comp": np.eye(N, dtype=complex), "fourier": dft(N), "random": Q.conj().T}
    rhos = [rho_sys(phi, N, s) for s in range(N)]
    max_fi = 0.0
    for B in bases.values():
        ps = np.array([np.real(np.diag(B @ r @ B.conj().T)) for r in rhos])
        for a in range(N - 1):
            dp = ps[a + 1] - ps[a]
            p = np.clip(ps[a], 1e-15, None)
            max_fi = max(max_fi, float(np.sum(dp * dp / p)))
    return max_fi


def trace_distance(ra: np.ndarray, rb: np.ndarray) -> float:
    d = ra - rb
    ev = np.linalg.eigvalsh((d + d.conj().T) / 2)
    return 0.5 * float(np.sum(np.abs(ev)))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main():
    groups = [4, 8, 16, 7]      # Z4/Z8/Z16 + prime-order Z7 (DLP-underlying cyclic)
    n_phi = 25
    rows = []
    print("LEG LB -- Zero-Fisher-Information no-go certificate")
    print("=" * 72)

    g_disc = g_ana = g_qfi = g_dth = g_cfi = g_td = 0.0
    for N in groups:
        A = shift_generator(N)
        rng = np.random.default_rng(1000 + N)
        m_disc = m_ana = m_qfi = m_dth = m_cfi = m_td = 0.0
        for _ in range(n_phi):
            phi = rand_state(N, rng)
            r0 = rho_sys(phi, N, 0)
            m_ana = max(m_ana, float(np.linalg.norm(r0 - twirl_analytic(phi, N))))
            # discrete indistinguishability over ALL shift pairs (s, s')
            rlist = [rho_sys(phi, N, s) for s in range(N)]
            for s in range(N):
                m_disc = max(m_disc, float(np.linalg.norm(rlist[s] - r0)))
                for sp in range(s + 1, N):
                    m_td = max(m_td, trace_distance(rlist[s], rlist[sp]))
            q, dth = qfi_and_dtheta(phi, N, A)
            m_qfi = max(m_qfi, q)
            m_dth = max(m_dth, dth)
            m_cfi = max(m_cfi, classical_fi_max(phi, N))

        rows.append(dict(
            group=f"Z{N}", N=N, prime=int(N == 7), n_phi=n_phi,
            max_discrete_frob=m_disc,
            max_trace_distance=m_td,
            max_num_vs_analytic=m_ana,
            max_classical_fi=m_cfi,
            max_qfi_theta=m_qfi,
            max_ddtheta_frob=m_dth,
        ))
        g_disc = max(g_disc, m_disc); g_ana = max(g_ana, m_ana)
        g_qfi = max(g_qfi, m_qfi); g_dth = max(g_dth, m_dth)
        g_cfi = max(g_cfi, m_cfi); g_td = max(g_td, m_td)
        print(f"Z{N:<3}{'(prime)' if N==7 else '':7} | discrete<= {m_disc:.2e} | "
              f"trace-dist<= {m_td:.2e} | num==analytic {m_ana:.2e} | "
              f"classFI<= {m_cfi:.2e} | QFI<= {m_qfi:.2e} | "
              f"||dRho/dtheta||<= {m_dth:.2e}")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "lb_cert.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print("=" * 72)
    print(f"GLOBAL: discrete<= {g_disc:.2e}  trace-dist<= {g_td:.2e}  "
          f"num==analytic<= {g_ana:.2e}")
    print(f"        classFI<= {g_cfi:.2e}  QFI<= {g_qfi:.2e}  "
          f"||dRho/dtheta||<= {g_dth:.2e}")
    checks = {
        "discrete<1e-12": g_disc < 1e-12,
        "trace_dist<1e-12": g_td < 1e-12,
        "num_vs_analytic<1e-12": g_ana < 1e-12,
        "classFI<1e-12": g_cfi < 1e-12,
        "QFI<1e-10": g_qfi < 1e-10,
        "ddtheta<1e-10": g_dth < 1e-10,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    print(f"CERTIFICATE: {verdict}", checks)
    print("wrote", path)


if __name__ == "__main__":
    main()
