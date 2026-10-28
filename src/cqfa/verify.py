"""Numerical verification of the invariance identities.

Re-derives the exact-invariance and coherent-surplus statements from the
package modules using fixed seeds. Importable, returning the computed
quantities, and runnable, printing a summary.
"""
from __future__ import annotations

import numpy as np

from . import group as G
from . import frame as L


def _rng():
    return np.random.default_rng(0)


def _rstate(rng, d):
    v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    return v / np.linalg.norm(v)


def _rherm(rng, d):
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    return (A + A.conj().T) / 2.0


def verify_equivariance(rep: G.Representation, n_inputs: int = 100) -> dict:
    """Theorem 1: FA and CQFA exactly invariant; base model not."""
    rng = _rng()
    d = rep.dim
    M = _rherm(rng, d)
    eb = ef = ec = 0.0
    for _ in range(n_inputs):
        psi = _rstate(rng, d)
        b0 = L.base_readout(psi, M)
        f0 = L.classical_fa_readout(psi, rep, M)
        c0 = L.cqfa_readout(psi, rep, M)[0]
        for U in rep.matrices:
            gx = U @ psi
            eb = max(eb, abs(L.base_readout(gx, M) - b0))
            ef = max(ef, abs(L.classical_fa_readout(gx, rep, M) - f0))
            c1 = L.cqfa_readout(gx, rep, M)[0]
            if c1 is not None and c0 is not None:
                ec = max(ec, abs(c1 - c0))
    return dict(name=rep.name, order=rep.order, dim=d,
                err_base=eb, err_fa=ef, err_cqfa=ec)


def verify_separation(rep: G.Representation, n_inputs: int = 2000) -> dict:
    """Theorem 2: symmetric-input surplus ~0; generic surplus >> 0."""
    rng = _rng()
    d = rep.dim
    M = _rherm(rng, d)
    Pi = rep.projector()
    sym = 0.0
    for _ in range(200):
        s = Pi @ _rstate(rng, d)
        if np.linalg.norm(s) < 1e-9:
            continue
        s = s / np.linalg.norm(s)
        sym = max(sym, abs(L.cqfa_readout(s, rep, M)[0] - L.classical_fa_readout(s, rep, M)))
    gen = 0.0
    for _ in range(n_inputs):
        psi = _rstate(rng, d)
        c = L.cqfa_readout(psi, rep, M)[0]
        if c is not None:
            gen = max(gen, abs(c - L.classical_fa_readout(psi, rep, M)))
    return dict(name=rep.name, surplus_symmetric=sym, surplus_generic=gen)


def main():
    reps = [G.z2_rc(), G.cyclic(4), G.symmetric(3), G.dihedral4()]
    print("=" * 70)
    print("CQFA framework verification (Theorems 1 & 2)")
    print("=" * 70)
    print("\nTheorem 1 — exact any-group invariance")
    for rep in reps:
        r = verify_equivariance(rep)
        print(f"  {r['name']:8s} |G|={r['order']} dim={r['dim']} | "
              f"FA={r['err_fa']:.1e} CQFA={r['err_cqfa']:.1e} base={r['err_base']:.2f}")
    print("\nTheorem 2 — coherent-vs-classical separation")
    for rep in (G.z2_pauli(3), G.cyclic(4), G.dihedral4()):
        r = verify_separation(rep)
        print(f"  {r['name']:10s} symmetric={r['surplus_symmetric']:.1e} (=>0)  "
              f"generic={r['surplus_generic']:.3f} (>>0)")
    print("=" * 70)


if __name__ == "__main__":
    main()
