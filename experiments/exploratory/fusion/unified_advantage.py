
from __future__ import annotations
import argparse, csv, os
import numpy as np

from experiments.spt_measurement_cost import readout_shadow as S

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def _op(d, n):
    o = np.array([[1.0 + 0j]])
    for i in range(n):
        o = np.kron(o, d.get(i, I2))
    return o


def _rstate(dim, m, rng):
    v = rng.standard_normal((m, dim)) + 1j * rng.standard_normal((m, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def run(sizes, seeds, out_dir, n_states=800):
    rows = []
    for n in sizes:
        A = _op({i: X for i in range(n)}, n)          # Z2 generator X^{(x)n}
        Mm = _op({0: Z}, n)                           # M = Z_0
        # real coherence operator: Re<psi|M A|psi> = <psi| (MA+AM)/2 |psi>  == <psi| Y_0 X_1..X_{n-1} |psi>
        Coh = (Mm @ A + A @ Mm) / 2.0
        label = {0: "Y", **{i: "X" for i in range(1, n)}}
        w = len(label)

        # (b) single-copy shadow variance ~ 3^w
        Vsh = []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            psi = _rstate(2 ** n, 1, rng)[0]
            nsnap = int(min(200000, 50 * 3 ** w))
            Vsh.append(float(np.var(S._per_snapshot_values(S.shadow_snapshots(psi, rng, nsnap), label))))
        V_shadow = float(np.mean(Vsh))

        # (a) classical FA cannot represent the coherence: train logreg on the diagonal
        # features (d_I=<M>, d_A=<AMA>) vs. the coherent feature; label = sign(Re C_IA).
        rng = np.random.default_rng(1234 + n)
        P = _rstate(2 ** n, n_states, rng)
        dI = np.real(np.einsum("mi,ij,mj->m", P.conj(), Mm, P))
        dA = np.real(np.einsum("mi,ij,mj->m", P.conj(), (A @ Mm @ A), P))
        coh = np.real(np.einsum("mi,ij,mj->m", P.conj(), Coh, P))
        y = (coh > np.median(coh)).astype(int)
        cut = n_states // 2

        def _logacc(Xf):
            Xf = np.atleast_2d(Xf.T).T if Xf.ndim == 1 else Xf
            mu, sd = Xf[:cut].mean(0), Xf[:cut].std(0) + 1e-9
            Xn = (Xf - mu) / sd; Xn = np.hstack([Xn, np.ones((len(Xn), 1))])
            w_ = np.zeros(Xn.shape[1])
            for _ in range(2000):
                g = 1 / (1 + np.exp(-(Xn[:cut] @ w_)))
                w_ -= 0.2 * (Xn[:cut].T @ (g - y[:cut]) / cut + 1e-3 * w_)
            pr = (1 / (1 + np.exp(-(Xn[cut:] @ w_)))) > 0.5
            return float(np.mean(pr.astype(int) == y[cut:]))

        acc_clfa = _logacc(np.stack([dI, dA], 1))     # classical FA: diagonal features
        acc_coh = _logacc(coh.reshape(-1, 1))          # coherent: the coherence feature
        rows.append(dict(n=n, weight=w, three_pow_w=3 ** w, V_shadow=V_shadow,
                         V_shadow_over_3w=V_shadow / 3 ** w, V_coherent=1.0,
                         classical_fa_acc=acc_clfa, coherent_acc=acc_coh,
                         shots_single_copy=V_shadow / 0.01, shots_coherent=1.0 / 0.01))
        print(f"n={n} w={w}: V_shadow/3^w={V_shadow/3**w:.2f} (single-copy 3^w) | "
              f"classical-FA acc={acc_clfa:.3f} vs coherent acc={acc_coh:.3f} | coherent O(1)")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "unified_advantage.csv")
    with open(p, "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w_.writeheader(); w_.writerows(rows)
    print("wrote", p)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/fusion")
    a = ap.parse_args(argv)
    run(a.sizes, a.seeds, a.out)


if __name__ == "__main__":
    main()
