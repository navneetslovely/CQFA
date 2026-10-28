
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

_ROOT = os.path.abspath(__file__)
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(
        os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from cqfa.group import cyclic, dihedral4, symmetric, z2_pauli  # noqa: E402

RNG = np.random.default_rng(20260721)
TOL = 1e-9


def induced_map(rep, alpha, beta):
    """Physically build |Phi> = O (sum_g beta_g|g>) (x) |psi>, post-select onto
    sum_g alpha_g |g>, and return the resulting linear map on the data register."""
    d = rep.dim
    m = rep.order
    dimF = 2 ** rep.n_frame_qubits
    a = np.zeros(dimF, dtype=complex)
    b = np.zeros(dimF, dtype=complex)
    a[:m] = alpha
    b[:m] = beta
    O = rep.controlled_oracle()
    M = np.zeros((d, d), dtype=complex)
    for j in range(d):
        psi = np.zeros(d, dtype=complex)
        psi[j] = 1.0
        phi = O @ np.kron(b, psi)
        col = np.zeros(d, dtype=complex)
        for f in range(dimF):
            col += np.conj(a[f]) * phi[f * d:(f + 1) * d]
        M[:, j] = col
    return M


def analytic_map(rep, alpha, beta):
    return sum(np.conj(alpha[g]) * beta[g] * rep.matrices[g] for g in range(rep.order))


def rand_c(n):
    return RNG.normal(size=n) + 1j * RNG.normal(size=n)


def check(name, rep, abelian):
    m, d = rep.order, rep.dim
    rows = []

    # R1: physical frame post-selection == group-algebra element, to ~1e-6
    errs = []
    for _ in range(20):
        al, be = rand_c(m), rand_c(m)
        errs.append(np.max(np.abs(induced_map(rep, al, be) - analytic_map(rep, al, be))))
    rows.append(("R1_physical_eq_group_algebra", float(np.max(errs))))

    # sanity: the uniform frame reproduces the fixed CQFA projector Pi
    u = np.ones(m) / np.sqrt(m)
    rows.append(("R1b_uniform_frame_eq_Pi",
                 float(np.max(np.abs(induced_map(rep, u, u) - rep.projector())))))

    # R2: achievable set == span_C{U_g}. Products conj(alpha_g)*beta_g range over all
    # of C^m (take alpha = e_g), so the reachable maps span exactly the group algebra.
    basis = np.array([rep.matrices[g].reshape(-1) for g in range(m)])
    reach = []
    for g in range(m):
        e = np.zeros(m, dtype=complex)
        e[g] = 1.0
        reach.append(induced_map(rep, e, np.ones(m, dtype=complex)).reshape(-1))
    reach = np.array(reach)
    r_alg = np.linalg.matrix_rank(basis, tol=1e-8)
    r_reach = np.linalg.matrix_rank(np.vstack([basis, reach]), tol=1e-8)
    rows.append(("R2_dim_group_algebra", float(r_alg)))
    rows.append(("R2_dim_reachable_union", float(r_reach)))  # == r_alg means no gain

    # R3 (abelian): span_C{U_g} == span_C{irrep projectors}
    if abelian:
        # characters of Z_m from the regular-type rep: diagonalise a generator
        w, V = np.linalg.eig(rep.matrices[1] if m > 1 else rep.matrices[0])
        projs = []
        seen = []
        for k in range(len(w)):
            if any(abs(w[k] - s) < 1e-8 for s in seen):
                continue
            seen.append(w[k])
            idx = [j for j in range(len(w)) if abs(w[j] - w[k]) < 1e-8]
            Vi = V[:, idx]
            projs.append((Vi @ np.linalg.pinv(Vi)).reshape(-1))
        projs = np.array(projs)
        r_p = np.linalg.matrix_rank(np.vstack([basis, projs]), tol=1e-8)
        rows.append(("R3_dim_algebra_plus_irrep_projectors", float(r_p)))

    # R4: subgroup indicator c_g = 1_H(g)/|H| is inside the reachable set, and the
    # induced map equals the fixed-frame CQFA projector for that subgroup.
    sub_err = np.nan
    for size in range(1, m):
        for S in itertools.combinations(range(m), size):
            if 0 not in S:
                continue
            prods = [rep.matrices[i] @ rep.matrices[j] for i in S for j in S]
            closed = all(any(np.max(np.abs(P - rep.matrices[k])) < 1e-9 for k in S) for P in prods)
            if not closed or size == 1:
                continue
            c = np.zeros(m, dtype=complex)
            c[list(S)] = 1.0 / size
            PiH = sum(c[g] * rep.matrices[g] for g in range(m))
            al = np.ones(m, dtype=complex)
            sub_err = float(np.max(np.abs(induced_map(rep, al, c) - PiH)))
            break
        if not np.isnan(sub_err):
            break
    rows.append(("R4_subgroup_projector_reachable_err", sub_err))

    return [(name, m, d, k, v) for k, v in rows]


def main():
    reps = [("cyclic4", cyclic(4), True),
            ("cyclic6", cyclic(6), True),
            ("z2_pauli3", z2_pauli(3), True),
            ("dihedral4", dihedral4(), False),
            ("symmetric3", symmetric(3), False)]
    out = []
    for name, rep, ab in reps:
        out += check(name, rep, ab)

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, "results", "c2_learned_frame_reduction.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("group,order,dim,quantity,value\n")
        for r in out:
            f.write("%s,%d,%d,%s,%.12g\n" % r)
    for r in out:
        print("%-11s |G|=%-2d d=%-2d %-38s %.3e" % r)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
