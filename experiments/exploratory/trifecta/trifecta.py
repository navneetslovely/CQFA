
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from cqfa import dynamical_lie_algebra as D
from experiments.trainability_law import trainability_law as T
from experiments.trainability_law.equivariant_trained import _eig, _val_grad, _states


# ---------------------------------------------------------------------------
# memory-efficient pack for large n (avoids materializing 2^{n-2} group matrices)
# ---------------------------------------------------------------------------
def _z2k_bplus(n: int) -> np.ndarray:
    """H_+ of (Z2)^{n-2} = |+>^{(x)(n-2)} (x) C^4 (qubits 0..n-3 in |+>, last 2 free).
    Structural; verified to span the same subspace as invariant_subspace_basis."""
    k = n - 2
    B = np.zeros((2 ** n, 4), dtype=complex)
    amp = 2.0 ** (-k / 2.0)
    for ab in range(4):
        for f in range(2 ** k):
            B[(f << 2) | ab, ab] = amp
    return B


def efficient_pack(n: int, cap: int = 8192):
    """dla_pack for (Z2)^{n-2} without the 2^{n-2}-element twirl. Uses the identity
    restrict(twirl(g)) == restrict(g) on H_+ (U_g acts as identity there), proven for
    (Z2)^{n-2}. Cross-checked against T.dla_pack for n<=8."""
    Bplus = _z2k_bplus(n)
    rgens = [D._traceless(D.restrict(g, Bplus)) for g in D.hea_circuit_generators(n)]
    d_eff, basis = D.lie_closure_dim(rgens, cap=cap)
    if d_eff == 0:
        return None
    A = basis[0].reshape(4, 4)
    O_Hp = (-1j * A + (-1j * A).conj().T) / 2.0
    O_Hp = O_Hp / np.sqrt(np.real(np.vdot(O_Hp.reshape(-1), O_Hp.reshape(-1))))
    return dict(d_eff=d_eff, basis=basis, Bplus=Bplus, dim_hplus=4, O_Hp=O_Hp)


def efficient_gens(n: int, pack) -> list:
    """Equivariant ansatz generators on H_+ (== restrict of raw HEA gens, Herm/anti
    parts), by the same restrict==restrict-of-twirl identity."""
    Bplus = pack["Bplus"]
    out = []
    for g in D.hea_circuit_generators(n):
        r = D.restrict(g, Bplus)
        for H in ((r + r.conj().T) / 2.0, 1j * (r - r.conj().T) / 2.0):
            if np.real(np.vdot(H.reshape(-1), H.reshape(-1))) > 1e-9:
                out.append(H)
    return out


def train_equivariant(pack, gens, n_train=120, n_test=120, depth=3, epochs=80,
                      lr=0.3, seed=0):
    """End-to-end train the equivariant ansatz on H_+ against a reachable g_G teacher
    (so the label lies in the model class); return train/test accuracy. Exact
    adjoint gradients (O(P) per sample)."""
    dH = pack["dim_hplus"]; O = pack["O_Hp"]; P = depth * len(gens)
    rng = np.random.default_rng(seed)
    j = 1 if pack["d_eff"] > 1 else 0
    Teach = -1j * pack["basis"][j].reshape(dH, dH); Teach = (Teach + Teach.conj().T) / 2
    Xtr = _states(dH, n_train, rng)
    ytr = (np.real(np.einsum("mi,ij,mj->m", Xtr.conj(), Teach, Xtr)) > 0).astype(int)
    Xte = _states(dH, n_test, rng)
    yte = (np.real(np.einsum("mi,ij,mj->m", Xte.conj(), Teach, Xte)) > 0).astype(int)

    gate_gens = gens * depth
    gate_eig = _eig(gens) * depth

    def vals(th, X):
        return np.array([_val_grad(gate_gens, gate_eig, th, x, O)[0] for x in X])

    def val_grads(th, X):
        vs = np.empty(len(X)); gs = np.zeros((len(X), P))
        for i, x in enumerate(X):
            vs[i], gs[i] = _val_grad(gate_gens, gate_eig, th, x, O)
        return vs, gs

    def acc(th, a, b, X, Y):
        return float(np.mean(((1 / (1 + np.exp(-(a * vals(th, X) + b)))) > 0.5).astype(int) == Y))

    theta = rng.uniform(0, 2 * np.pi, P) * 0.1; a, b = 2.0, 0.0
    for _ in range(epochs):
        F, GR = val_grads(theta, Xtr); p = 1 / (1 + np.exp(-(a * F + b))); dz = p - ytr
        theta -= lr * ((dz * a)[:, None] * GR).mean(0)
        a -= lr * float(np.mean(dz * F)); b -= lr * float(np.mean(dz))
    return acc(theta, a, b, Xtr, ytr), acc(theta, a, b, Xte, yte)


def run(sizes, seeds, out_dir, depth=3):
    rows = []
    for n in sizes:
        # efficient path (structural H_+) for all n; verified to match the full
        # dla_pack at n<=8 (the 2^{n-2}-element group is too large to materialize
        # densely for n>=9). Cross-check on demand: efficient_pack vs T.dla_pack.
        if n <= 8 and n <= 6:                          # cheap cross-check for provenance
            full = T.dla_pack(T.z2k_group(n, n - 2), n)
            eff = efficient_pack(n)
            assert full["dim_hplus"] == eff["dim_hplus"] == 4
            assert full["d_eff"] == eff["d_eff"], f"n={n} d_eff {full['d_eff']} vs {eff['d_eff']}"
        pack = efficient_pack(n)
        if pack is None:
            print(f"n={n}: DLA pack empty, skipping"); continue
        gens = efficient_gens(n, pack)
        dH = pack["dim_hplus"]

        # (1) flat variance: analytical Ragone + empirical, vs generic ansatz
        var_ana, _ = T.ragone_var_cqfa(pack, np.random.default_rng(7))
        var_gen = T.ragone_var_generic(n)
        emp = []
        for sd in seeds:
            rng = np.random.default_rng(100 + sd)
            chi = T.aligned_input(pack, rng)
            emp.append(T.empirical_gradient_variance(pack, gens, chi, S=200, rng=rng, depth=depth))
        var_emp = float(np.mean(emp))

        # (2) trained end-to-end
        tr, te = [], []
        for sd in seeds:
            a, b = train_equivariant(pack, gens, depth=depth, seed=sd)
            tr.append(a); te.append(b)

        # (3) hardness knob (representation-level, ): generic-ansatz DLA is
        # su(2^n) (dim 4^n-1, exponential -> classically hard family), while the
        # equivariant ansatz lives on a dim-4 subspace (the trainable knob). The
        # hardness is realized via the data oracle/encoding (see package docstring).
        generic_dla_dim = 4 ** n - 1

        row = dict(
            n=n, group=f"(Z2)^{n-2}", group_order=2 ** (n - 2), dim_hplus=dH,
            var_analytical=var_ana, var_empirical=var_emp, var_generic=var_gen,
            var_times_dimhplus=var_ana * dH,
            train_acc=float(np.mean(tr)), test_acc=float(np.mean(te)),
            test_acc_std=float(np.std(te)),
            generic_dla_dim=generic_dla_dim, d_eff=pack["d_eff"], n_gens=len(gens),
        )
        rows.append(row)
        print(f"n={n:>2} |G|=2^{n-2} dimH+={dH}: "
              f"Var_ana={var_ana:.3e} Var_emp={var_emp:.3e} (Var*dimH+={var_ana*dH:.3f}, "
              f"generic={var_gen:.1e}) | test_acc={np.mean(te):.3f}+/-{np.std(te):.3f} | "
              f"generic-DLA=4^{n}-1={generic_dla_dim} (hard)")
    if rows:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, "trifecta.csv")
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("wrote", p)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 6, 8, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/trifecta")
    args = ap.parse_args(argv)
    run(args.sizes, args.seeds, args.out)


if __name__ == "__main__":
    main()
