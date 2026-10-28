
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from cqfa import group as G
from cqfa import dynamical_lie_algebra as D


FIELDS = ["which", "n", "dim", "dim_g_ambient", "dim_g_restricted",
          "dim_Hplus", "theory_su_dim", "capped", "cap", "wall_s"]


def compute_row(n: int, cap: int) -> list:
    rows = []
    theory = 4 ** n - 1
    dim = 2 ** n

    # ambient circuit DLA of the generic HEA (should be su(2^n))
    t0 = time.time()
    gens = D.hea_circuit_generators(n, include_entangler=True)
    d_amb, _ = D.lie_closure_dim(gens, cap=cap)
    capped = d_amb >= cap
    wall_amb = time.time() - t0

    # frame-induced restricted DLA g_G on H_+ (z2_pauli(n) family)
    t1 = time.time()
    rep = G.z2_pauli(n)
    d_res, _, Bplus = D.frame_induced_dla(rep, n, cap=cap)
    dim_hplus = int(Bplus.shape[1])
    wall_res = time.time() - t1

    rows.append(dict(which="generic_hea", n=n, dim=dim,
                     dim_g_ambient=int(d_amb), dim_g_restricted="",
                     dim_Hplus="", theory_su_dim=theory,
                     capped=bool(capped), cap=cap, wall_s=round(wall_amb, 4)))
    rows.append(dict(which="cqfa_restricted", n=n, dim=dim,
                     dim_g_ambient="", dim_g_restricted=int(d_res),
                     dim_Hplus=dim_hplus, theory_su_dim=theory,
                     capped=False, cap=cap, wall_s=round(wall_res, 4)))
    return rows


def main():
    ap = argparse.ArgumentParser(description="DLA-dimension sweep")
    ap.add_argument("--n-min", type=int, default=2)
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--cap", type=int, default=4096,
                    help="hard cap on the ambient DLA basis (memory guard)")
    ap.add_argument("--out", type=str,
                    default="experiments/trainability_law/out/dla.csv")
    args = ap.parse_args()

    print(f"[dla] numpy {np.__version__}  n={args.n_min}..{args.n_max}  cap={args.cap}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    all_rows = []
    for n in range(args.n_min, args.n_max + 1):
        rows = compute_row(n, args.cap)
        for r in rows:
            all_rows.append(r)
            print(f"  n={n:>2} {r['which']:<16} "
                  f"ambient={r['dim_g_ambient']} restricted={r['dim_g_restricted']} "
                  f"|H+|={r['dim_Hplus']} (4^n-1={r['theory_su_dim']}) "
                  f"capped={r['capped']} {r['wall_s']}s")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    manifest = {"script": "run_dla", "numpy": np.__version__,
                "args": vars(args), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "n_rows": len(all_rows)}
    with open(os.path.splitext(args.out)[0] + "_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[dla] wrote {args.out} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
