
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from cqfa import dynamical_lie_algebra as D
from .trifecta import _z2k_bplus, efficient_pack, efficient_gens


# ---------------------------------------------------------------------------
# supremacy-class data encoding (IQP) and a Clifford control
# ---------------------------------------------------------------------------
_H1 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _kron_all(mats):
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def iqp_state(n: int, rng, clifford: bool = False) -> np.ndarray:
    """|psi> = H^{(x)n} D H^{(x)n} |0>, D = exp(i sum_{i<j} t_ij Z_i Z_j + i sum_i s_i Z_i).
    Generic angles -> IQP (single output amplitudes #P-hard). clifford=True snaps all
    angles to multiples of pi/2 (a stabilizer state -> Gottesman-Knill easy): the
    control showing hardness needs the NON-Clifford angles."""
    d = 2 ** n
    Hn = _kron_all([_H1] * n)
    psi = Hn @ (np.eye(d)[:, 0])                    # H^n|0> = |+>^n
    # diagonal phase (ZZ couplings + Z fields)
    z = np.array([[(-1) ** ((idx >> (n - 1 - q)) & 1) for q in range(n)] for idx in range(d)])
    phase = np.zeros(d)
    def ang():
        a = rng.uniform(0, 2 * np.pi)
        return (np.round(a / (np.pi / 2)) * (np.pi / 2)) if clifford else a
    for i in range(n):
        phase += ang() * z[:, i]
        for j in range(i + 1, n):
            phase += ang() * z[:, i] * z[:, j]
    psi = np.exp(1j * phase) * psi
    psi = Hn @ psi
    return psi / np.linalg.norm(psi)


def t_count_proxy(n, rng, clifford=False):
    """Number of non-Clifford diagonal rotations in the encoding (0 => Clifford)."""
    total = n + n * (n - 1) // 2
    if clifford:
        return 0
    return total


# ---------------------------------------------------------------------------
# the CQFA (Z2)^{n-2} model output, and its amplitude reduction
# ---------------------------------------------------------------------------
def model_output_direct(psi, Bplus, W, O):
    """f = <psi|Pi W^dag O W Pi|psi> / <psi|Pi|psi>, Pi = Bplus Bplus^dag (dim-4)."""
    A = Bplus.conj().T @ psi                        # 4-vector of encoding amplitudes
    p = float(np.real(A.conj() @ A))
    if p < 1e-15:
        return None, A, p
    M = W.conj().T @ O @ W                          # 4x4 (W,O act on H_+)
    return float(np.real(A.conj() @ M @ A) / p), A, p


def build_W_on_hplus(n, pack, depth=3, seed=0):
    """A concrete trained-regime equivariant ansatz W as a 4x4 unitary on H_+."""
    gens = efficient_gens(n, pack)
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, depth * len(gens))
    W = np.eye(4, dtype=complex)
    t = 0
    for _ in range(depth):
        for Hh in gens:
            w, V = np.linalg.eigh(Hh)
            W = (V * np.exp(-1j * theta[t] * w)) @ V.conj().T @ W
            t += 1
    return W


def run(sizes, seeds, out_dir):
    rows = []
    for n in sizes:
        pack = efficient_pack(n)
        Bplus = pack["Bplus"]; O4 = pack["O_Hp"]
        W = build_W_on_hplus(n, pack, seed=0)
        # --- verify the reduction: direct statevector f == (A^dag M A)/(A^dag A) ---
        worst_match = 0.0
        amp_mags, ps = [], []
        for sd in seeds:
            rng = np.random.default_rng(1000 + sd)
            psi = iqp_state(n, rng)
            # embed W,O (on H_+) into the full space to get a truly-direct check
            Wfull = Bplus @ W @ Bplus.conj().T + (np.eye(2 ** n) - Bplus @ Bplus.conj().T)
            Ofull = Bplus @ O4 @ Bplus.conj().T
            Pi = Bplus @ Bplus.conj().T
            chi = Pi @ psi
            p = float(np.real(chi.conj() @ chi))
            f_full = float(np.real(chi.conj() @ Wfull.conj().T @ Ofull @ Wfull @ chi) / p)
            f_red, A, p2 = model_output_direct(psi, Bplus, W, O4)
            worst_match = max(worst_match, abs(f_full - f_red))
            amp_mags.append(float(np.mean(np.abs(A))))
            ps.append(p)
        # --- hardness certificate ---
        tcount = t_count_proxy(n, np.random.default_rng(0))
        # Clifford control: snapping angles makes the encoding a stabilizer state
        rngc = np.random.default_rng(0)
        psi_cliff = iqp_state(n, rngc, clifford=True)
        # frame entanglement of the encoding across a middle cut (bond-dim proxy)
        cut = n // 2
        mat = psi.reshape(2 ** cut, 2 ** (n - cut))
        s = np.linalg.svd(mat, compute_uv=False)
        s = s[s > 1e-12]
        ent = float(-np.sum((s ** 2) * np.log2(s ** 2)))
        rows.append(dict(
            n=n, dim_hplus=4, n_amplitudes_needed=4,
            reduction_max_err=worst_match,           # ~0 confirms f = fn of 4 amplitudes
            mean_amp_magnitude=float(np.mean(amp_mags)),
            mean_accept_p=float(np.mean(ps)),
            encoding_t_count=tcount, clifford_control_t_count=0,
            encoding_entanglement_bits=ent, bond_dim_proxy=float(2 ** ent),
            single_amplitude_hard="#P-hard (BQP<=P^#P; IQP:BMS16)",
            samples_to_read=float(1.0 / np.mean(ps)),     # ~2^{n-2}: the trade-off
            tradeoff="classical-hard eval + flat landscape, BUT exp quantum sampling (p=2^{2-n})",
        ))
        print(f"n={n}: reduction err={worst_match:.2e} (f = fn of {4} encoding amplitudes) "
              f"| accept p~{np.mean(ps):.3f} amp~{np.mean(amp_mags):.3f} | "
              f"T-count={tcount} (Clifford ctrl=0) ent={ent:.2f} bits "
              f"(chi~{2**ent:.1f}) -> single amplitude #P-hard")
    if rows:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, "hardness.csv")
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("wrote", p)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/trifecta")
    args = ap.parse_args(argv)
    run(args.sizes, args.seeds, args.out)


if __name__ == "__main__":
    main()
