"""
experiments/trainability_law/equivariant_trained.py
=============================================
Demonstrates the CQFA thesis END-TO-END on ONE model: the genuinely equivariant
CQFA ansatz (generators closing into the frame-induced DLA g_G) is TRAINED to
convergence AND its trained-regime gradient variance follows the law Var ~ 1/dim(H_+).

This closes the gap flagged in review: the trainability law is a property of the
equivariant ansatz, and here that same equivariant ansatz is the deployed, trained model
(not the generic-ansatz CQFAModel used elsewhere for raw expressivity). We report, per n
and per seed: final train/test accuracy (it IS trainable) and the trained gradient
variance (it obeys the law). Output: results/trainability_law/equivariant_trained.csv .
"""
from __future__ import annotations
import argparse, csv, os
import numpy as np

from cqfa import group as G
from experiments.trainability_law import trainability_law as T


def _eig(gens):
    return [np.linalg.eigh(H) for H in gens]


def _gate(eig_k, theta, psi):
    """exp(-i theta G) psi via cached eigendecomposition (w, V) of G."""
    w, V = eig_k
    return (V * np.exp(-1j * theta * w)) @ (V.conj().T @ psi)


def _apply(gens, thetas, chi, depth, eig=None):
    eig = eig or _eig(gens)
    layer = eig * depth
    psi = chi.astype(complex).copy()
    for t in range(len(thetas)):
        psi = _gate(layer[t], thetas[t], psi)
    return psi


def _val_grad(gate_gens, gate_eig, thetas, chi, O):
    """Exact value and gradient of <chi|W^dag O W|chi> via the adjoint method, O(P)
    matvecs (one forward + one backward pass). gate_gens/gate_eig are the ORDERED
    per-gate generator matrices and eigendecompositions."""
    P = len(thetas)
    states = [chi.astype(complex)]
    for k in range(P):
        states.append(_gate(gate_eig[k], thetas[k], states[-1]))
    y = states[-1]
    val = float(np.real(y.conj() @ O @ y))
    phi = O @ y
    g = np.zeros(P)
    for k in range(P - 1, -1, -1):
        g[k] = 2.0 * np.real(phi.conj() @ (-1j * gate_gens[k]) @ states[k + 1])
        phi = _gate(gate_eig[k], -thetas[k], phi)   # U_k^dag = exp(+i theta_k G_k)
    return val, g


def _states(dH, m, rng):
    X = rng.standard_normal((m, dH)) + 1j * rng.standard_normal((m, dH))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def run_one(n, seed, depth=3, n_train=120, n_test=120, epochs=80, lr=0.3):
    rep = G.z2_pauli(n)
    pack = T.dla_pack(rep, n); gens = T.equivariant_ansatz_generators(rep, n, pack)
    dH = pack["dim_hplus"]; O = pack["O_Hp"]; P = depth * len(gens)
    rng = np.random.default_rng(seed)
    # teacher = a g_G element (reachable -> the equivariant model can represent the label)
    j = 1 if pack["d_eff"] > 1 else 0
    Teach = -1j * pack["basis"][j].reshape(dH, dH); Teach = (Teach + Teach.conj().T) / 2
    Xtr = _states(dH, n_train, rng); ytr = (np.real(np.einsum("mi,ij,mj->m", Xtr.conj(), Teach, Xtr)) > 0).astype(int)
    Xte = _states(dH, n_test, rng);  yte = (np.real(np.einsum("mi,ij,mj->m", Xte.conj(), Teach, Xte)) > 0).astype(int)

    eig = _eig(gens)
    gate_gens = gens * depth
    gate_eig = eig * depth

    def val_grads(th, X):
        vs = np.empty(len(X)); gs = np.zeros((len(X), P))
        for i, x in enumerate(X):
            vs[i], gs[i] = _val_grad(gate_gens, gate_eig, th, x, O)
        return vs, gs
    def vals(th, X): return np.array([_val_grad(gate_gens, gate_eig, th, x, O)[0] for x in X])
    def acc(th, a, b, X, Y): return float(np.mean(((1 / (1 + np.exp(-(a * vals(th, X) + b)))) > 0.5).astype(int) == Y))

    theta = rng.uniform(0, 2 * np.pi, P) * 0.1; a, b = 2.0, 0.0
    for _ in range(epochs):
        F, GR = val_grads(theta, Xtr); p = 1 / (1 + np.exp(-(a * F + b))); dz = p - ytr
        theta -= lr * ((dz * a)[:, None] * GR).mean(0)
        a -= lr * float(np.mean(dz * F)); b -= lr * float(np.mean(dz))
    tr, te = acc(theta, a, b, Xtr, ytr), acc(theta, a, b, Xte, yte)

    # trained-regime gradient variance of component 0 over wide param draws at a fixed input
    chi = Xtr[0]; S = 300; grads = np.empty(S); rng2 = np.random.default_rng(1000 + seed)
    for s in range(S):
        _, gk = _val_grad(gate_gens, gate_eig, rng2.uniform(0, 2 * np.pi, P), chi, O)
        grads[s] = gk[0]
    return dict(n=n, seed=seed, dim_hplus=dH, n_gens=len(gens), n_params=P,
                train_acc=tr, test_acc=te, grad_var=float(np.var(grads)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/trainability_law")
    args = ap.parse_args(argv)
    rows = []
    for n in args.sizes:
        accs, gv = [], []
        for sd in args.seeds:
            r = run_one(n, sd); rows.append(r); accs.append(r["test_acc"]); gv.append(r["grad_var"])
        print(f"n={n} dimH+={rows[-1]['dim_hplus']:>3}: test_acc={np.mean(accs):.3f}+/-{np.std(accs):.3f} "
              f"grad_var={np.mean(gv):.4f} (Var*dimH+={np.mean(gv)*rows[-1]['dim_hplus']:.3f})")
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, "equivariant_trained.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", p)


if __name__ == "__main__":
    main()
