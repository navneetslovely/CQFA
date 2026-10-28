"""
experiments.trainability_law.run_certificate
======================================
The Certified Representation Dial (CRD): compute the (T, H) certificate points
for each model on the z2_pauli(n) family and write two CSVs.

  certificate_points.csv   -- one row per (rep, n, model, seed): the raw axes
                              T_d_eff, T_var_lb, T_P_g_rho, T_P_g_M and
                              H_escape, H_surplus, H_frame_entropy, H_bond_chi.
  certificate_scaling.csv  -- one row per (family, model): DLA-dim poly/exp fit
                              of T, mean H at n_max, and the assigned quadrant.

For the non_equivariant model the trainability coordinate is read from the
AMBIENT su(2^n) DLA (dim 4^n - 1) -- the barren case -- rather than the
frame-induced restricted DLA, so the dial correctly places it in the
untrainable quadrant.

 ledger (see README.md):
  * T mechanism (Ragone-Fontana law + poly dim(g_G)) : PROVEN for the special
    case (z2_pauli / RC-EQC), EMPIRICAL for the general family.
  * H1 escape flag : UNCONDITIONAL under its own (non-re-preparable data) premise.
  * H2 proxies (surplus, frame entropy / bond dim) : EVIDENCE, a resource
    witness, NOT a classical-hardness theorem.

Usage:

    python -m experiments.trainability_law.run_certificate \
        --n-min 2 --n-max 5 --models cqfa classical_fa non_equivariant \
        --source quantum_nonreprepable --n-data 64 --seeds 0 1 2 3 4 \
        --out experiments/trainability_law/out
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from cqfa import group as G
from cqfa import dynamical_lie_algebra as D
from cqfa import frame as L
from cqfa.certificate import (trainability_axis, hardness_axis,
                              reachable_observable, QUANTUM, CLASSICAL)
from . import analysis as A


POINT_FIELDS = ["rep", "n", "n_frame", "model", "seed", "source",
                "T_d_eff", "T_dim_hplus", "T_var_lb", "T_P_g_rho", "T_P_g_M",
                "H_escape", "H_surplus", "H_frame_entropy", "H_bond_chi",
                "p_norm"]
SCALE_FIELDS = ["family", "model", "n_min", "n_max",
                "d_eff_at_nmax", "d_eff_poly_R2", "d_eff_exp_R2", "poly_vs_exp",
                "T_scalar_at_nmax", "H_frame_entropy_at_nmax", "H_escape",
                "H_surplus_mean", "quadrant"]


def _ambient_trainability(rep, n, chi, M):
    """Trainability axis for the generic HEA: ambient su(2^n) DLA + raw chi.

    This is the barren case; d_eff = 4^n - 1 so var_lb collapses exponentially.
    """
    gens = D.hea_circuit_generators(n)
    d_amb, basis = D.lie_closure_dim(gens, cap=4 ** n - 1)
    rho = np.outer(chi, chi.conj())
    P_g_rho = D.g_purity(rho, basis)
    P_g_M = D.g_purity(M, basis)
    return dict(d_eff=int(d_amb), dim_hplus=int(rep.dim),
                var_lb=(P_g_rho * P_g_M / d_amb if d_amb else 0.0),
                P_g_rho=P_g_rho, P_g_M=P_g_M)


def compute_points(args):
    rows = []
    ns = list(range(args.n_min, args.n_max + 1))
    for n in ns:
        rep = G.z2_pauli(n)
        M, ok = reachable_observable(rep, n)
        for model_name in args.models:
            for seed in args.seeds:
                rng = np.random.default_rng(1000 * seed + n)
                # average H over n_data draws; T is deterministic per state via chi
                surpluses, entropies, plps = [], [], []
                T = None
                for _ in range(args.n_data):
                    psi = A.random_pure_state(rep.dim, rng)
                    phi, p = L.coherent_projection(psi, rep)
                    chi = phi / np.sqrt(p) if p > 1e-12 else phi
                    H = hardness_axis(rep, psi, M, args.source)
                    surpluses.append(H["surplus"])
                    entropies.append(H["frame_entropy"])
                    plps.append(H["p_norm"])
                    if T is None:  # T from the first valid chi
                        if model_name == "non_equivariant":
                            T = _ambient_trainability(rep, n, psi, M)
                        else:
                            T = trainability_axis(rep, n, chi, M)
                escape = 1 if args.source == QUANTUM else 0
                rows.append(dict(
                    rep=rep.name, n=n, n_frame=rep.n_frame_qubits,
                    model=model_name, seed=seed, source=args.source,
                    T_d_eff=T["d_eff"], T_dim_hplus=T["dim_hplus"],
                    T_var_lb=T["var_lb"], T_P_g_rho=T["P_g_rho"],
                    T_P_g_M=T["P_g_M"], H_escape=escape,
                    H_surplus=float(np.mean(surpluses)),
                    H_frame_entropy=float(np.mean(entropies)),
                    H_bond_chi=float(np.mean([2.0 ** e for e in entropies])),
                    p_norm=float(np.mean(plps))))
            print(f"  n={n:>2} {model_name:<16} d_eff={T['d_eff']:>5} "
                  f"var_lb={T['var_lb']:.3e} "
                  f"S_frame={np.mean(entropies):.3f} escape={escape}")
    return ns, rows


def _quadrant(poly_vs_exp, H_escape, H_surplus):
    # Trainability is determined by the DLA SCALING (the mechanism), not an
    # absolute var_lb cutoff: poly dim(g_G) => no barren plateau => trainable;
    # exp dim(g) (su(2^n)) => barren plateau => untrainable.  This is n-robust,
    # unlike thresholding var_lb whose absolute value shrinks for every model.
    trainable = (poly_vs_exp == "poly")
    hard = (H_escape == 1) and (H_surplus > 1e-6)
    if trainable and hard:
        return "trainable_and_hard"
    if trainable and not hard:
        return "trainable_and_simulable"
    if (not trainable) and hard:
        return "hard_and_untrainable"
    return "untrainable_and_simulable"


def compute_scaling(ns, rows, models):
    scale_rows = []
    for model_name in models:
        # across-seed mean d_eff and T per n
        d_eff_by_n, T_by_n, S_by_n, surp_by_n, esc = [], [], [], [], 0
        for n in ns:
            sub = [r for r in rows if r["model"] == model_name and r["n"] == n]
            if not sub:
                continue
            d_eff_by_n.append(np.mean([r["T_d_eff"] for r in sub]))
            vlb = np.mean([r["T_var_lb"] for r in sub])
            T_by_n.append(np.log10(vlb) if vlb > 0 else -np.inf)
            S_by_n.append(np.mean([r["H_frame_entropy"] for r in sub]))
            surp_by_n.append(np.mean([r["H_surplus"] for r in sub]))
            esc = int(round(np.mean([r["H_escape"] for r in sub])))
        d_eff_by_n = np.array(d_eff_by_n, dtype=float)
        # poly vs exp fit of dim(g) itself
        c, _, r2_poly = A.fit_poly(ns, d_eff_by_n)     # log dim vs log n
        b, _, r2_exp = A.fit_exp(ns, d_eff_by_n)       # log dim vs n
        poly_vs_exp = "poly" if r2_poly >= r2_exp else "exp"
        T_nmax = T_by_n[-1]
        quad = _quadrant(poly_vs_exp, esc, surp_by_n[-1])
        scale_rows.append(dict(
            family="z2_pauli", model=model_name, n_min=ns[0], n_max=ns[-1],
            d_eff_at_nmax=float(d_eff_by_n[-1]), d_eff_poly_R2=r2_poly,
            d_eff_exp_R2=r2_exp, poly_vs_exp=poly_vs_exp,
            T_scalar_at_nmax=float(T_nmax),
            H_frame_entropy_at_nmax=float(S_by_n[-1]), H_escape=esc,
            H_surplus_mean=float(np.mean(surp_by_n)), quadrant=quad))
    return scale_rows


def main():
    ap = argparse.ArgumentParser(description="Certified Representation Dial")
    ap.add_argument("--n-min", type=int, default=2)
    ap.add_argument("--n-max", type=int, default=5)
    ap.add_argument("--models", type=str, nargs="+",
                    default=["cqfa", "classical_fa", "non_equivariant"])
    ap.add_argument("--source", type=str, default=QUANTUM,
                    choices=[QUANTUM, CLASSICAL])
    ap.add_argument("--n-data", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=str, default="experiments/trainability_law/out")
    args = ap.parse_args()

    print(f"[certificate] numpy {np.__version__}  n={args.n_min}..{args.n_max}  "
          f"source={args.source}  n_data={args.n_data}  seeds={args.seeds}")
    os.makedirs(os.path.abspath(args.out), exist_ok=True)

    ns, rows = compute_points(args)
    scale_rows = compute_scaling(ns, rows, args.models)

    pts_path = os.path.join(args.out, "certificate_points.csv")
    scl_path = os.path.join(args.out, "certificate_scaling.csv")
    with open(pts_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=POINT_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(scl_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SCALE_FIELDS)
        w.writeheader()
        for r in scale_rows:
            w.writerow(r)

    manifest = {"script": "run_certificate", "numpy": np.__version__,
                "args": vars(args), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "n_point_rows": len(rows), "n_scale_rows": len(scale_rows)}
    with open(os.path.join(args.out, "certificate_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[certificate] wrote {pts_path} ({len(rows)} rows) and "
          f"{scl_path} ({len(scale_rows)} rows)")
    for r in scale_rows:
        print(f"  [{r['model']:<16}] d_eff(nmax)={r['d_eff_at_nmax']:.0f} "
              f"({r['poly_vs_exp']}) T={r['T_scalar_at_nmax']:.2f} "
              f"S={r['H_frame_entropy_at_nmax']:.2f} -> {r['quadrant']}")


if __name__ == "__main__":
    main()
