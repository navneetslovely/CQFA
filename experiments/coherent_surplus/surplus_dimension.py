"""
experiments/coherent_surplus/surplus_dimension.py
==========================================
The coherent-surplus EXPRESSIVITY theorem, quantified.

A CQFA feature is a quadratic form <psi|A|psi> whose operator is a combination of the
group coherences  C_{gh} = U_g^dag M U_h  (g,h in G). Classical frame averaging sees
only the DIAGONAL coherences {C_{gg}} (the orbit outputs), a space of dimension |G|.
The coherent readout sees ALL coherences, a space of dimension min(|G|^2, d^2) (d =
rep dimension). Hence:

    coherent-surplus dimension  =  min(|G|^2, d^2) - |G| .

In the faithful/regular regime (|G| <= d) this is |G|(|G|-1): the coherent model's
expressivity grows QUADRATICALLY in |G| while classical frame averaging grows only
LINEARLY. Verified below across Z2, C_n, S_3, D_4 (surplus matches the closed form
exactly). Output: results/coherent_surplus/surplus_dimension.csv .
"""
from __future__ import annotations
import argparse, csv, os
import numpy as np

from cqfa import group as G


def _herm_parts(A):
    return [(A + A.conj().T) / 2, (A - A.conj().T) / (2j)]


def _rank(ops, tol=1e-9):
    return int(np.linalg.matrix_rank(np.array([o.reshape(-1) for o in ops]), tol=tol))


def surplus_for(rep, seed=0):
    Us = rep.matrices; m = len(Us); d = rep.dim
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)); M = (M + M.conj().T) / 2
    diag = [Us[g].conj().T @ M @ Us[g] for g in range(m)]
    allops = []
    for g in range(m):
        for h in range(m):
            allops += _herm_parts(Us[g].conj().T @ M @ Us[h])
    r_classical = _rank(diag)
    r_coherent = _rank(allops)
    return dict(name=rep.name, order=m, dim=d,
                classical_dim=r_classical, coherent_dim=r_coherent,
                surplus_dim=r_coherent - r_classical,
                closed_form=min(m * m, d * d) - m)


def run(out_dir):
    reps = [G.z2_rc(), G.z2_pauli(3)] + [G.cyclic(n) for n in (3, 4, 5, 6, 7, 8)] \
        + [G.symmetric(3), G.dihedral4()]
    rows = []
    print(f"{'group':>8}{'|G|':>5}{'d':>4}{'classical':>10}{'coherent':>9}{'surplus':>8}{'closed-form':>12}{'ok':>4}")
    for rep in reps:
        r = surplus_for(rep)
        r["matches"] = (r["surplus_dim"] == r["closed_form"])
        rows.append(r)
        print(f"{r['name']:>8}{r['order']:>5}{r['dim']:>4}{r['classical_dim']:>10}"
              f"{r['coherent_dim']:>9}{r['surplus_dim']:>8}{r['closed_form']:>12}"
              f"{'Y' if r['matches'] else 'N':>4}")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "surplus_dimension.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", p, "| all match closed form:", all(r["matches"] for r in rows))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/coherent_surplus")
    run(ap.parse_args().out)
