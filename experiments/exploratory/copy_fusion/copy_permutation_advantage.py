
from __future__ import annotations
import argparse, csv, os
import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
_EIG = {"X": np.linalg.eigh(X)[1], "Y": np.linalg.eigh(Y)[1], "Z": np.linalg.eigh(Z)[1]}
_NAMES = ["X", "Y", "Z"]


def _kron(ms):
    o = np.array([[1.0 + 0j]])
    for m in ms:
        o = np.kron(o, m)
    return o


def _swap(n):
    d = 2 ** n
    S = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            S[j * d + i, i * d + j] = 1.0
    return S


def _rand_pure(n, rng):
    v = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
    return v / np.linalg.norm(v)


def _depolarize(psi, n, p):
    """rho = (1-p)|psi><psi| + p I/2^n : purity 1 -> ... as p grows."""
    d = 2 ** n
    return (1 - p) * np.outer(psi, psi.conj()) + p * np.eye(d) / d


def _shadow_rho_hat(rho, n, rng):
    """One single-copy random-Pauli classical-shadow snapshot rho_hat (Huang-Kueng-Preskill)."""
    bases = rng.integers(0, 3, n)
    U = _kron([_EIG[_NAMES[b]].conj().T for b in bases])
    probs = np.real(np.diag(U @ rho @ U.conj().T)); probs = np.clip(probs, 0, None); probs /= probs.sum()
    out = rng.choice(len(probs), p=probs)
    bits = [(out >> (n - 1 - k)) & 1 for k in range(n)]
    snap = np.array([[1.0 + 0j]])
    for k in range(n):
        v = _EIG[_NAMES[bases[k]]][:, bits[k]].reshape(-1, 1)
        snap = np.kron(snap, 3 * (v @ v.conj().T) - I2)
    return snap


def run(sizes, seeds, out_dir):
    rows = []
    for n in sizes:
        SW = _swap(n)
        idres, shvar, s2var = [], [], []
        for sd in seeds:
            rng = np.random.default_rng(sd + n)
            psi = _rand_pure(n, rng)
            rho = np.outer(psi, psi.conj())
            # (a) identity <rho x rho|SWAP> = Tr(rho^2)
            rr = np.kron(rho, rho)
            idres.append(abs(np.real(np.trace(SW @ rr)) - np.real(np.trace(rho @ rho))))
            # (b) coherent S_2 (SWAP) per-shot variance for purity: outcome +-1, mean=Tr(rho^2)
            pur = float(np.real(np.trace(rho @ rho)))
            s2var.append(1 - pur ** 2)
            # (c) single-copy shadow purity U-statistic variance
            K = 60
            snaps = [_shadow_rho_hat(rho, n, rng) for _ in range(K)]
            ests = [np.real(np.trace(snaps[a] @ snaps[b])) for a in range(K) for b in range(a + 1, K)]
            shvar.append(float(np.var(ests)))
        eps = 0.1
        V_sh = float(np.mean(shvar)); V_s2 = max(float(np.mean(s2var)), 1e-6)
        rows.append(dict(n=n, coherence_eq_purity_err=float(np.mean(idres)),
                         V_coherent_swap=V_s2, V_singlecopy_shadow=V_sh,
                         copies_coherent=max(V_s2 / eps ** 2, 1.0),
                         copies_singlecopy=V_sh / eps ** 2,
                         separation_ratio=(V_sh / eps ** 2) / max(V_s2 / eps ** 2, 1.0)))
        print(f"n={n}: |<SWAP>-Tr(rho^2)|={np.mean(idres):.1e} | coherent V={V_s2:.3f} (O(1)) | "
              f"single-copy V={V_sh:.1f} (~exp) | ratio={rows[-1]['separation_ratio']:.1e}")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "copy_permutation_advantage.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", p)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/copy_fusion")
    a = ap.parse_args(argv)
    run(a.sizes, a.seeds, a.out)


if __name__ == "__main__":
    main()
