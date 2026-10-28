
from __future__ import annotations
import argparse, csv, os
import numpy as np

from cqfa import group as G
from cqfa import dynamical_lie_algebra as D
from cqfa.dynamical_lie_algebra import (symmetrized_generators, hea_circuit_generators, restrict,
                      frame_induced_dla, g_purity)

X = np.array([[0, 1], [1, 0]], dtype=complex)
I2 = np.eye(2, dtype=complex)


# ---------------------------------------------------------------------------
# groups: (Z2)^k as k INDEPENDENT single-qubit flips on n qubits -> dim H_+ = 2^(n-k)
# ---------------------------------------------------------------------------
def _opX(qubits, n):
    o = np.array([[1.0 + 0j]])
    for i in range(n):
        o = np.kron(o, X if i in qubits else I2)
    return o


def z2k_group(n: int, k: int) -> G.Representation:
    """(Z2)^k on n qubits: X on each of qubits 0..k-1 (independent). |G|=2^k,
    dim(H_+) = 2^(n-k). Built directly (no close_matrices)."""
    elems = []
    for mask in range(2 ** k):
        S = {i for i in range(k) if (mask >> i) & 1}
        elems.append(_opX(S, n))
    # identity first
    elems.sort(key=lambda M: 0 if np.allclose(M, np.eye(2 ** n)) else 1)
    return G.Representation(f"Z2^{k}", elems, 2 ** n)


# ---------------------------------------------------------------------------
# DLA + reachable observable, computed ONCE per config (avoids the double call)
# ---------------------------------------------------------------------------
def dla_pack(rep, n, cap=8192):
    d_eff, basis, Bplus = frame_induced_dla(rep, n, cap=cap)
    dim_hplus = Bplus.shape[1]
    if d_eff == 0:
        return None
    # observable = a g_G basis element (Hermitian on H_+), P_g(O)=1 by construction
    A = basis[0].reshape(dim_hplus, dim_hplus)
    O_Hp = (-1j * A + (-1j * A).conj().T) / 2.0
    O_Hp = O_Hp / np.sqrt(np.real(np.vdot(O_Hp.reshape(-1), O_Hp.reshape(-1))))  # HS-normalise
    return dict(d_eff=d_eff, basis=basis, Bplus=Bplus, dim_hplus=dim_hplus, O_Hp=O_Hp)


def aligned_input(pack, rng):
    """A state on H_+ aligned with g_G: top eigenvector of a random g_G generator."""
    b = pack["basis"]
    j = rng.integers(0, b.shape[0])
    A = b[j].reshape(pack["dim_hplus"], pack["dim_hplus"])
    H = (-1j * A + (-1j * A).conj().T) / 2.0
    w, V = np.linalg.eigh(H)
    return V[:, -1]


def ragone_var_cqfa(pack, rng, n_inputs=8):
    """Analytical Ragone variance for the equivariant CQFA ansatz, averaged over
    aligned inputs.  Var = P_g(rho) P_g(O) / dim(g)."""
    PgO = g_purity(pack["O_Hp"], pack["basis"])
    vals = []
    for _ in range(n_inputs):
        chi = aligned_input(pack, rng)
        rho = np.outer(chi, chi.conj())
        vals.append(g_purity(rho, pack["basis"]) * PgO / pack["d_eff"])
    return float(np.mean(vals)), float(np.std(vals))


def ragone_var_generic(n):
    """Generic ansatz g = su(2^n): Var = P_su(rho) P_su(O)/dim(su) ~ 1/4^n."""
    return (1.0 - 1.0 / 2 ** n) * 1.0 / (4 ** n - 1)


# ---------------------------------------------------------------------------
# (B) EMPIRICAL gradient-variance sampling of a genuinely equivariant ansatz
# ---------------------------------------------------------------------------
def equivariant_ansatz_generators(rep, n, pack):
   
    Bplus = pack["Bplus"]
    gens = symmetrized_generators(hea_circuit_generators(n), rep)
    out = []
    for g in gens:
        r = restrict(g, Bplus)
        herm = (r + r.conj().T) / 2.0               # Hermitian part
        anti = (r - r.conj().T) / 2.0               # anti-Hermitian part -> i*anti is Hermitian
        for H in (herm, 1j * anti):                  # both are valid Hermitian generators
            if np.real(np.vdot(H.reshape(-1), H.reshape(-1))) > 1e-9:
                out.append(H)
    return out


def _expm_herm(theta, H, cache):
    """exp(-i theta H) via cached eigendecomposition of H."""
    key = id(H)
    if key not in cache:
        cache[key] = np.linalg.eigh(H)
    w, V = cache[key]
    return (V * np.exp(-1j * theta * w)) @ V.conj().T


def empirical_gradient_variance(pack, gens, chi, S, rng, depth=3, eps=1e-4):
   
    O = pack["O_Hp"]
    layer = list(gens)
    L = len(layer)
    cache = {}

    def W_apply(thetas):
        psi = chi.astype(complex).copy()
        t = 0
        for _ in range(depth):
            for H in layer:
                psi = _expm_herm(thetas[t], H, cache) @ psi
                t += 1
        return psi

    P = depth * L
    grads = np.empty(S)
    for s in range(S):
        th = rng.uniform(0, 2 * np.pi, P)
        thp = th.copy(); thp[0] += eps
        thm = th.copy(); thm[0] -= eps
        yp = W_apply(thp); ym = W_apply(thm)
        fp = np.real(yp.conj() @ O @ yp)
        fm = np.real(ym.conj() @ O @ ym)
        grads[s] = (fp - fm) / (2 * eps)
    return float(np.var(grads))


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------
def _group_for(tag, n, k):
    
    if tag == "milder_plateau":
        return G.z2_pauli(n)                       # X^{(x)n}: dim H_+ = 2^(n-1), poly DLA
    return z2k_group(n, k)                          # (Z2)^k independent flips: dim H_+ = 2^(n-k)


def run_scaling(seeds, out_dir):
    
    rows = []
    configs = []
    
    for n in range(2, 11):
        configs.append(("milder_plateau", n, 1))
    
    for c in (2, 3):
        for n in range(c + 1, 9):        # start at the true minimum n (k = n-c >= 1)
            configs.append(("escape", n, n - c))
    for tag, n, k in configs:
        rep = _group_for(tag, n, k)
        pack = dla_pack(rep, n)
        if pack is None:
            continue
        vc, sc = [], []
        for sd in seeds:
            m, s = ragone_var_cqfa(pack, np.random.default_rng(1000 + sd))
            vc.append(m)
        var_cqfa = float(np.mean(vc)); var_cqfa_sd = float(np.std(vc))
        var_gen = ragone_var_generic(n)
        fam = "Z2-global" if tag == "milder_plateau" else f"Z2^{k}(dimH+={pack['dim_hplus']})"
        rows.append(dict(experiment=tag, family=fam, n=n, k=k,
                         dim_full=2 ** n, dim_hplus=pack["dim_hplus"], dim_gG=pack["d_eff"],
                         var_cqfa=var_cqfa, var_cqfa_std=var_cqfa_sd, var_generic=var_gen,
                         ratio=var_cqfa / var_gen,
                         var_cqfa_times_dimHplus=var_cqfa * pack["dim_hplus"],
                         var_generic_times_4n=var_gen * (4 ** n)))
        print(f"[{tag:8s}] Z2^{k} n={n} dimH+={pack['dim_hplus']:>4} dim_gG={pack['d_eff']:>3} "
              f"Var_cqfa={var_cqfa:.3e} Var_gen={var_gen:.3e} ratio={var_cqfa/var_gen:.1f} "
              f"Var*dimH+={var_cqfa*pack['dim_hplus']:.4f}")
    _write(os.path.join(out_dir, "trainability_law.csv"), rows)
    return rows


def run_empirical(seeds, out_dir, S=400, depth=3):
  
    rows = []
    for n in range(2, 9):                              # Z2 global-flip (poly DLA): validate the law
        rep = G.z2_pauli(n)
        pack = dla_pack(rep, n)
        if pack is None:
            continue
        if True:
            gens = equivariant_ansatz_generators(rep, n, pack)
            if not gens:
                continue
            emp = []
            for sd in seeds:
                rng = np.random.default_rng(2000 + sd)
                chi = aligned_input(pack, rng)
                emp.append(empirical_gradient_variance(pack, gens, chi, S, rng, depth=depth))
            var_emp = float(np.mean(emp)); var_emp_sd = float(np.std(emp))
            m, _ = ragone_var_cqfa(pack, np.random.default_rng(0))
            rows.append(dict(family="Z2-global", n=n, dim_hplus=pack["dim_hplus"],
                             dim_gG=pack["d_eff"], n_gens=len(gens), depth=depth, S=S,
                             var_empirical=var_emp, var_empirical_std=var_emp_sd,
                             var_ragone=m, var_generic=ragone_var_generic(n)))
            print(f"[empirical] Z2-global n={n} dimH+={pack['dim_hplus']:>4} "
                  f"Var_emp={var_emp:.3e}+/-{var_emp_sd:.1e} Var_ragone={m:.3e} "
                  f"Var_gen={ragone_var_generic(n):.3e}")
    _write(os.path.join(out_dir, "trainability_empirical.csv"), rows)
    return rows


def _write(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, f"({len(rows)} rows)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/trainability_law")
    ap.add_argument("--empirical", action="store_true", help="also run (B) empirical sampling")
    ap.add_argument("--S", type=int, default=400)
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args(argv)
    print("=== (A) scaling collapse + (C) escape [analytical Ragone] ===")
    run_scaling(args.seeds, args.out)
    if args.empirical:
        print("\n=== (B) empirical equivariant-ansatz gradient sampling ===")
        run_empirical(args.seeds, args.out, S=args.S, depth=args.depth)


if __name__ == "__main__":
    main()
