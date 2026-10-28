"""
Complex-irrep-safe certification that F_G is a generalized Fourier transform.

WHY THIS FILE EXISTS
--------------------
u1_structure_check.csv ships gft_verified=0 for five of twelve groups -- A4, F21 and
ALL THREE abelian controls C6, C8, C12 -- because irrep_dims_from_centre() cannot
split complex irreps. It builds a HERMITIAN generic element of the group-algebra
centre and calls eigh(). For a complex-conjugate pair of irreps the Hermitian part
symmetrises the two characters onto the SAME eigenvalue, so their isotypic components
merge into one eigenspace of dimension d^2 + d^2. For A4 (irreps 1,1,1,3) the two
conjugate 1-dimensional irreps merge into a dimension-2 eigenspace, 2 is not a perfect
square, and the assertion trips: "isotypic dims [1,2,9] are not perfect squares".
F21 = C7:C3 fails identically. The abelian controls were never reached because the
sweep aborted first.

The failure is in the DIAGNOSTIC, not in F_G. This module certifies the same property
by a route that has no eigendecomposition in it at all, so complex irreps are a
non-issue.

METHOD -- pure sparsity structure, no eigenvalues
-------------------------------------------------
F is a generalized Fourier transform for G iff F^dag L_g F is block-diagonal, with the
SAME block structure, for every g in G. The regular representation decomposes as
    L  ~  (+)_lambda  d_lambda copies of lambda,
so the block multiset is: each irrep lambda contributes d_lambda blocks of size
d_lambda. We therefore:

  1. accumulate the support of |F^dag L_g F| over ALL g (not a few random states, so an
     accidental cancellation cannot merge two blocks), and take connected components;
  2. certify block-diagonality directly: max |entry| strictly outside those blocks;
  3. read the irrep dimensions back OUT of the block multiset. If a block size d occurs
     n_d times then the number of irreps of dimension d is n_d / d, which must be a
     positive integer -- a strong consistency check that a wrong F cannot pass;
  4. verify the two theorems that must then hold:
         sum_lambda d_lambda^2 == |G|      and      #irreps == #conjugacy classes.

Steps 3 and 4 are what make this a certification rather than a description: a matrix
that merely happens to be block-diagonal will fail the integrality of n_d/d or one of
the two counting identities.

PRE-REGISTERED, TWO-SIDED: F_G is CERTIFIED for a group iff block-diagonality holds to
< 1e-9, every n_d/d is a positive integer, sum d^2 == |G|, and #irreps == #classes.
It is REFUTED if any of those fails -- which would mean the Mode-F readout used
throughout U1 is not a Fourier readout for that group, and every U1 number for it would
have to be withdrawn. This is a real possibility for A4 and F21 precisely because they
are the groups the shipped diagnostic could not check.

Reads : nothing (regenerates the bases from the group tables)
Writes: results/readout_basis/u1_gft_certification.csv   (NEW; nothing overwritten)
"""
from __future__ import annotations

import csv
import math
import os

import numpy as np

import experiments.readout_basis.u1_shotmatched_nonabelian as U

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "results", "readout_basis", "u1_gft_certification.csv")

GROUPS = ["S3", "D4", "Q8", "D5", "A4", "D6", "D10", "F21", "S4", "C6", "C8", "C12"]
TOL = 1e-9

# published irrep dimensions, used ONLY as an independent cross-check of what the
# structural method reads off; the certification itself never consults them.
KNOWN = {"S3": [1, 1, 2], "D4": [1, 1, 1, 1, 2], "Q8": [1, 1, 1, 1, 2],
         "D5": [1, 1, 2, 2], "A4": [1, 1, 1, 3], "D6": [1, 1, 1, 1, 2, 2],
         "D10": [1, 1, 1, 1, 2, 2, 2, 2], "F21": [1, 1, 1, 3, 3],
         "S4": [1, 1, 2, 3, 3], "C6": [1] * 6, "C8": [1] * 8, "C12": [1] * 12}


def components(adj):
    """Connected components of a boolean adjacency matrix -> list of index sets."""
    n = adj.shape[0]
    seen, comps = set(), []
    for i in range(n):
        if i in seen:
            continue
        stack, comp = [i], set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack.extend(int(v) for v in np.where(adj[u])[0] if v not in comp)
        seen |= comp
        comps.append(sorted(comp))
    return comps


def certify(name):
    T, m = U.group_table(name)
    L, Rop = U.regular_ops(T)
    F = U.fourier_basis(L, Rop, U.CANONICAL_BASIS)
    n_classes = len(U.conjugacy_classes(T))

    # 1. support of F^dag L_g F accumulated over EVERY g
    adj = np.zeros((m, m), dtype=bool)
    worst_block = 0.0
    Bs = []
    for g in range(m):
        B = F.conj().T @ L[g] @ F
        Bs.append(B)
        adj |= (np.abs(B) > TOL)
    adj |= adj.T
    comps = components(adj)

    # 2. block-diagonality: largest magnitude strictly OUTSIDE the blocks
    mask = np.ones((m, m), dtype=bool)
    for c in comps:
        idx = np.ix_(c, c)
        mask[idx] = False
    for B in Bs:
        worst_block = max(worst_block, float(np.max(np.abs(B[mask]))) if mask.any() else 0.0)

    # 3. read irrep dimensions back out of the block multiset
    sizes = sorted(len(c) for c in comps)
    counts = {}
    for s in sizes:
        counts[s] = counts.get(s, 0) + 1
    dims, integral = [], True
    for d, n_d in sorted(counts.items()):
        q = n_d / d
        if abs(q - round(q)) > 1e-12 or round(q) < 1:
            integral = False
        else:
            dims.extend([d] * int(round(q)))
    dims = sorted(dims)

    # 4. the two counting theorems
    sum_dsq = sum(d * d for d in dims)
    ok_blocks = worst_block < TOL
    ok_sum = (sum_dsq == m)
    ok_classes = (len(dims) == n_classes)
    certified = bool(ok_blocks and integral and ok_sum and ok_classes)

    matches_known = (dims == sorted(KNOWN[name]))
    return dict(
        group=name, order=m, abelian=int(U.is_abelian(T)),
        n_conjugacy_classes=n_classes,
        block_sizes="|".join(map(str, sizes)),
        max_offblock_magnitude=worst_block,
        blockdiag_ok=int(ok_blocks),
        multiplicity_integral=int(integral),
        irrep_dims_recovered="|".join(map(str, dims)) if dims else "none",
        sum_d_squared=sum_dsq, sum_d_squared_equals_order=int(ok_sum),
        n_irreps=len(dims), n_irreps_equals_n_classes=int(ok_classes),
        gft_certified=int(certified),
        irrep_dims_published="|".join(map(str, sorted(KNOWN[name]))),
        matches_published=int(matches_known),
        method="structural_blockdiag_no_eigendecomposition",
    )


def main():
    rows = [certify(g) for g in GROUPS]
    assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"{'group':>5} {'|G|':>4} {'ab':>3} {'blocks':>22} {'offblock':>10} "
          f"{'dims recovered':>16} {'sum d^2':>8} {'#irr=#cls':>10} {'CERT':>5} {'=pub':>5}")
    for r in rows:
        print(f"{r['group']:>5} {r['order']:>4} {r['abelian']:>3} {r['block_sizes'][:22]:>22} "
              f"{r['max_offblock_magnitude']:10.2e} {r['irrep_dims_recovered'][:16]:>16} "
              f"{r['sum_d_squared']:>8} {r['n_irreps_equals_n_classes']:>10} "
              f"{'YES' if r['gft_certified'] else 'NO':>5} {'yes' if r['matches_published'] else 'NO':>5}")
    n = sum(r["gft_certified"] for r in rows)
    k = sum(r["matches_published"] for r in rows)
    print(f"\n  certified {n}/{len(rows)}   recovered dims match published {k}/{len(rows)}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
