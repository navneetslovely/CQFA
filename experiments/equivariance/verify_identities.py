
from __future__ import annotations

import argparse
import csv

import cqfa
from cqfa.paths import RESULTS, ensure

REPRESENTATIONS = [
    ("z2_rc", lambda: cqfa.group.z2_rc()),
    ("z2_pauli2", lambda: cqfa.group.z2_pauli(2)),
    ("z2_pauli4", lambda: cqfa.group.z2_pauli(4)),
    ("c4", lambda: cqfa.group.cyclic(4)),
    ("c6", lambda: cqfa.group.cyclic(6)),
    ("s3", lambda: cqfa.group.symmetric(3)),
    ("d4", lambda: cqfa.group.dihedral4()),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    rows = []
    for name, build in REPRESENTATIONS:
        try:
            rep = build()
        except Exception as exc:                      # representation unavailable
            print(f"  {name:<12} skipped: {exc}")
            continue
        worst_a = worst_b = worst_base = 0.0
        for seed in args.seeds:
            out = cqfa.verify.verify_equivariance(rep, seed=seed) \
                if "seed" in cqfa.verify.verify_equivariance.__code__.co_varnames \
                else cqfa.verify.verify_equivariance(rep)
            worst_a = max(worst_a, float(out["err_cqfa"]))
            worst_b = max(worst_b, float(out["err_fa"]))
            worst_base = max(worst_base, float(out["err_base"]))
        rows.append(dict(representation=rep.name, order=rep.order, dim=rep.dim,
                         n_frame_qubits=rep.n_frame_qubits, n_seeds=len(args.seeds),
                         max_err_mode_a=worst_a, max_err_mode_b=worst_b,
                         max_err_unsymmetrised=worst_base))
        print(f"  {rep.name:<12} Mode A {worst_a:.2e}   Mode B {worst_b:.2e}"
              f"   unsymmetrised control {worst_base:.3f}")

    out_dir = ensure(RESULTS / "equivariance") if args.out is None else ensure(
        __import__("pathlib").Path(args.out))
    path = out_dir / "invariance_identities.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
