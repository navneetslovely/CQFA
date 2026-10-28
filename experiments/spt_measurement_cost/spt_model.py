
from __future__ import annotations

import numpy as np

# single-qubit building blocks -------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def op_on(paulis_by_site: dict, n: int) -> np.ndarray:
    """Full 2^n operator from {site: 2x2 Pauli}; unspecified sites carry I."""
    out = np.array([[1.0 + 0j]])
    for i in range(n):
        out = np.kron(out, paulis_by_site.get(i, I2))
    return out


# -- the canonical Hamiltonian -------------------------------------------------
def cluster_ising_xx_hamiltonian(n: int, lam: float) -> np.ndarray:
    """H(lam) = -sum_i Z_{i-1} X_i Z_{i+1} + lam sum_i X_i X_{i+1}  (dense, 2^n x 2^n)."""
    if n < 3:
        raise ValueError("cluster-Ising model needs n >= 3 interior sites")
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(1, n - 1):
        H -= op_on({i - 1: Z, i: X, i + 1: Z}, n)
    for i in range(0, n - 1):
        H += lam * op_on({i: X, i + 1: X}, n)
    return H


def z2xz2_generators(n: int):
    """The Z2xZ2 protecting symmetry generators (P_even, P_odd) = (prod X_even, prod X_odd)."""
    p_even = op_on({i: X for i in range(0, n, 2)}, n)
    p_odd = op_on({i: X for i in range(1, n, 2)}, n)
    return p_even, p_odd


def z2xz2_commutator_max(n: int, lam: float) -> float:
    """max ||[H(lam), P]|| over the two Z2xZ2 generators (should be ~0)."""
    H = cluster_ising_xx_hamiltonian(n, lam)
    p_even, p_odd = z2xz2_generators(n)
    return float(max(np.max(np.abs(H @ p_even - p_even @ H)),
                     np.max(np.abs(H @ p_odd - p_odd @ H))))


# -- stabilizers and the string order parameter --------------------------------
def stabilizer(i: int, n: int) -> np.ndarray:
    """Interior cluster stabilizer K_i = Z_{i-1} X_i Z_{i+1} (1 <= i <= n-2)."""
    return op_on({i - 1: Z, i: X, i + 1: Z}, n)


def boundary_stabilizers(n: int):
    """The two open-chain boundary stabilizers (X_0 Z_1, Z_{n-2} X_{n-1})."""
    left = op_on({0: X, 1: Z}, n)
    right = op_on({n - 2: Z, n - 1: X}, n)
    return left, right


def string_operator(n: int, a: int | None = None, b: int | None = None) -> np.ndarray:
  
    if a is None:
        a = 1
    if b is None:
        b = n - 2
    O = np.eye(2 ** n, dtype=complex)
    for i in range(a, b + 1):
        O = stabilizer(i, n) @ O
    return O


def naive_string_operator(n: int) -> np.ndarray:
  
    d = {0: Z, n - 1: Z}
    for i in range(1, n - 1):
        d[i] = X
    return op_on(d, n)


def string_order(psi: np.ndarray, n: int, a: int | None = None, b: int | None = None) -> float:
    """S(a,b) = <psi| prod_{i=a}^{b} K_i |psi>, the SPT nonlocal order parameter."""
    O = string_operator(n, a, b)
    return float(np.real(psi.conj() @ O @ psi))


def local_z_mid(psi: np.ndarray, n: int) -> float:
    
    mid = n // 2
    return float(np.real(psi.conj() @ op_on({mid: Z}, n) @ psi))


# -- ground state extraction ---------------------------------------------------
def ground_state(n: int, lam: float, edge_pin: float = 0.0, sparse: bool = False) -> np.ndarray:
   
    if sparse:
        return _ground_state_sparse(n, lam, edge_pin)
    H = cluster_ising_xx_hamiltonian(n, lam)
    if edge_pin > 0.0:
        left, right = boundary_stabilizers(n)
        H = H - edge_pin * (left + right)
    w, v = np.linalg.eigh(H)
    return v[:, 0]


def _ground_state_sparse(n: int, lam: float, edge_pin: float) -> np.ndarray:
    """Sparse ground state for large n (optional). Statevector-only."""
    from scipy import sparse
    from scipy.sparse.linalg import eigsh

    sI = sparse.identity(2, format="csr", dtype=complex)
    sX = sparse.csr_matrix(X)
    sY = sparse.csr_matrix(Y)  # noqa: F841  (kept for parity / possible extension)
    sZ = sparse.csr_matrix(Z)

    def sop(paulis_by_site):
        out = sparse.identity(1, format="csr", dtype=complex)
        for i in range(n):
            out = sparse.kron(out, paulis_by_site.get(i, sI), format="csr")
        return out

    dim = 2 ** n
    H = sparse.csr_matrix((dim, dim), dtype=complex)
    for i in range(1, n - 1):
        H = H - sop({i - 1: sZ, i: sX, i + 1: sZ})
    for i in range(0, n - 1):
        H = H + lam * sop({i: sX, i + 1: sX})
    if edge_pin > 0.0:
        H = H - edge_pin * (sop({0: sX, 1: sZ}) + sop({n - 2: sZ, n - 1: sX}))
    w, v = eigsh(H, k=6, which="SA")
    return v[:, int(np.argmin(w))]


def cluster_state(n: int) -> np.ndarray:
    
    dim = 2 ** n
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0  # |0...0>
    stabs = []
    left, right = boundary_stabilizers(n)
    stabs.append(left)
    for i in range(1, n - 1):
        stabs.append(stabilizer(i, n))
    stabs.append(right)
    ident = np.eye(dim, dtype=complex)
    for S in stabs:
        psi = 0.5 * (ident + S) @ psi
    nrm = np.linalg.norm(psi)
    if nrm < 1e-12:
        raise RuntimeError("stabilizer projection annihilated |0..0>; check stabilizer signs")
    return psi / nrm


# -- class pools (the labelled learning task) ----------------------------------
SPT_LAMBDAS = (0.0, 0.2, 0.3, 0.4)      # class 1; 0.0 -> exact cluster state
TRIVIAL_LAMBDAS = (2.0, 2.5, 3.0)       # class 0
SPT_LABEL = 1
TRIVIAL_LABEL = 0


def prepare_state(n: int, lam: float, phase: str, edge_pin: float = 0.0,
                  sparse: bool = False) -> np.ndarray:
   
    if phase == "spt" and abs(lam) < 1e-12:
        return cluster_state(n)
    return ground_state(n, lam, edge_pin=edge_pin, sparse=sparse)


def sample_class_state(n: int, rng: np.random.Generator,
                       spt_lambdas=SPT_LAMBDAS, trivial_lambdas=TRIVIAL_LAMBDAS,
                       edge_pin: float = 0.0, sparse: bool = False):
   
    y = int(rng.integers(0, 2))
    if y == SPT_LABEL:
        lam = float(rng.choice(spt_lambdas))
        phase = "spt"
    else:
        lam = float(rng.choice(trivial_lambdas))
        phase = "trivial"
    psi = prepare_state(n, lam, phase, edge_pin=edge_pin, sparse=sparse)
    return psi, y, lam, phase


# -- self test -----------------------------------------------------------------
def _selftest() -> bool:
    ok = True
    print(f"{'n':>3}{'lam':>6}{'S':>10}{'Zmid':>10}{'comm':>10}")
    for n in (6, 8):
        for lam in (0.0, 0.3, 1.0, 2.5):
            if lam == 0.0:
                psi = cluster_state(n)
            else:
                psi = ground_state(n, lam)
            s = string_order(psi, n)
            print(f"{n:>3}{lam:>6.1f}{s:>10.4f}{local_z_mid(psi, n):>10.4f}"
                  f"{z2xz2_commutator_max(n, lam):>10.1e}")
        # gates
        s_clu = string_order(cluster_state(n), n)
        s_spt = string_order(ground_state(n, 0.3), n)
        s_tri = string_order(ground_state(n, 2.5), n)
        naive = float(np.real(cluster_state(n).conj() @ naive_string_operator(n)
                              @ cluster_state(n)))
        if abs(s_clu - 1.0) > 1e-9:
            ok = False
        if s_spt < 0.9 or abs(s_tri) > 0.1:
            ok = False
        if abs(naive) > 1e-9:
            ok = False
        print(f"    n={n}: cluster S={s_clu:.6f}  naive-string={naive:.2e}")
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    _selftest()
