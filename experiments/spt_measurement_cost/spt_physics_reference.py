
from __future__ import annotations
import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def op_on(paulis_by_site: dict, n: int) -> np.ndarray:
    """Full 2^n operator from {site: 2x2 Pauli}; unspecified sites carry I."""
    out = np.array([[1.0 + 0j]])
    for i in range(n):
        out = np.kron(out, paulis_by_site.get(i, I))
    return out


def cluster_ising_hamiltonian(n: int, h: float) -> np.ndarray:

    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(1, n - 1):
        H -= op_on({i - 1: Z, i: X, i + 1: Z}, n)
    for i in range(n):
        H -= h * op_on({i: X}, n)
    return H


def cluster_ising_xx_hamiltonian(n: int, lam: float) -> np.ndarray:
   
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(1, n - 1):
        H -= op_on({i - 1: Z, i: X, i + 1: Z}, n)
    for i in range(0, n - 1):
        H += lam * op_on({i: X, i + 1: X}, n)
    return H


def ground_state_xx(n: int, lam: float) -> np.ndarray:
    """Ground state of the canonical XX-competitor model."""
    w, v = np.linalg.eigh(cluster_ising_xx_hamiltonian(n, lam))
    return v[:, 0]


def ground_state(n: int, h: float) -> np.ndarray:
    """Exact ground state (unit vector) of the cluster-Ising model."""
    w, v = np.linalg.eigh(cluster_ising_hamiltonian(n, h))
    return v[:, 0]


def stabilizer(i: int, n: int) -> np.ndarray:
    """Interior cluster stabilizer K_i = Z_{i-1} X_i Z_{i+1}."""
    return op_on({i - 1: Z, i: X, i + 1: Z}, n)


def string_order(psi: np.ndarray, n: int, a: int | None = None, b: int | None = None) -> float:
    """S(a,b) = < prod_{i=a}^{b} K_i >, the SPT nonlocal order parameter."""
    if a is None:
        a = 1
    if b is None:
        b = n - 2
    O = np.eye(2 ** n, dtype=complex)
    for i in range(a, b + 1):
        O = stabilizer(i, n) @ O
    return float(np.real(psi.conj() @ O @ psi))


def local_mag_z(psi: np.ndarray, n: int) -> float:
    return float(np.mean([np.real(psi.conj() @ op_on({i: Z}, n) @ psi) for i in range(n)]))


def z2xz2_generators(n: int):
    """The Z2xZ2 protecting symmetry: (Xeven, Xodd)."""
    return op_on({i: X for i in range(0, n, 2)}, n), op_on({i: X for i in range(1, n, 2)}, n)


def _selftest():
    print(f"{'n':>3}{'h':>6}{'string_order':>14}{'|<Z>|avg':>10}")
    ok = True
    for n in (8, 10):
        for h in (0.0, 0.5, 0.9, 1.1, 2.0, 3.0):
            psi = ground_state(n, h)
            so = string_order(psi, n)
            print(f"{n:>3}{h:>6.1f}{so:>14.4f}{abs(local_mag_z(psi, n)):>10.4f}")
            if h == 0.0 and abs(so - 1.0) > 1e-6:
                ok = False
            if h == 3.0 and abs(so) > 1e-2:
                ok = False
        Xe, Xo = z2xz2_generators(n)
        H = cluster_ising_hamiltonian(n, 0.7)
        c = max(np.max(np.abs(H @ Xe - Xe @ H)), np.max(np.abs(H @ Xo - Xo @ H)))
        print(f"    n={n}: Z2xZ2 commutator max = {c:.2e}")
        if c > 1e-9:
            ok = False
        print("-" * 33)
    # canonical XX-competitor model: genuine transition + preserved symmetry
    print("XX-competitor model (canonical):")
    for n in (8, 10):
        Xe, Xo = z2xz2_generators(n)
        H = cluster_ising_xx_hamiltonian(n, 0.7)
        c = max(np.max(np.abs(H @ Xe - Xe @ H)), np.max(np.abs(H @ Xo - Xo @ H)))
        s_spt = string_order(ground_state_xx(n, 0.3), n)
        s_triv = string_order(ground_state_xx(n, 2.5), n)
        print(f"    n={n}: Z2xZ2 comm={c:.1e}  S(SPT,0.3)={s_spt:.3f}  S(trivial,2.5)={s_triv:.3f}")
        if c > 1e-9 or s_spt < 0.9 or abs(s_triv) > 0.1:
            ok = False
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    _selftest()
