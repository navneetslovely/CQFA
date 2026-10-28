"""
experiments.trainability_law.run_scaling
==================================
The gradient-variance-vs-qubit-count experiment (the "no-barren-plateau"
figure).  For each model and qubit count n it estimates

    V_m(n) = Var_theta[ d_mu f_m(theta; chi_fixed) ]

over S wide-random parameter draws (theta ~ U[0, 2pi)^P) at a FIXED input,
across >=5 outer seeds, with a per-point bootstrap 95% CI.  It then fits
log V vs n two ways (exponential vs polynomial) to classify each model as a
barren PLATEAU (exp decay) or TRAINABLE (poly / flat decay), and cross-checks
against the DLA Var ~ 1/dim(g) law.

Models:
  cqfa            -- coherent-projected input chi in H_+ (expected trainable).
  non_equivariant -- generic HEA on the raw state (expected barren plateau,
                     the negative control).
  classical_fa    -- orbit-output averaging (expected trainable).

The 'augmented' model is intentionally excluded (its forward pass and thus
per-sample gradient variance are identical to non_equivariant).

Deterministic; usage:

    python -m experiments.trainability_law.run_scaling \
        --n-min 2 --n-max 8 --samples 1000 --seeds 0 1 2 3 4 --n-layers 2 \
        --models cqfa non_equivariant classical_fa \
        --out experiments/trainability_law/out/variance.csv \
        --fits-out experiments/trainability_law/out/fits.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from cqfa import group as G
from cqfa import readout as Mo
from cqfa import dynamical_lie_algebra as D
from cqfa.certificate import reachable_observable
from . import analysis as A


VAR_FIELDS = ["model", "n", "dim", "n_layers", "n_params", "mu_kind", "mu_index",
              "samples", "seeds", "var_mean", "var_std", "var_ci_lo", "var_ci_hi",
              "var_pred", "p_norm_mean", "wall_s"]
FIT_FIELDS = ["model", "n_points", "mu_kind",
              "exp_slope_b", "exp_slope_b_ci_lo", "exp_slope_b_ci_hi", "r2_exp",
              "poly_exp_c", "poly_exp_c_ci_lo", "poly_exp_c_ci_hi", "r2_poly",
              "crosscheck_slope", "r2_crosscheck", "classification"]


def _build_model(model_name, rep, n_layers, observable):
    ctor = {"cqfa": Mo.CQFAModel, "classical_fa": Mo.ClassicalFAModel,
            "non_equivariant": Mo.NonEquivModel, "augmented": Mo.AugmentedModel}
    return ctor[model_name](rep, n_layers, observable=observable)


def _dla_prediction(model_name, rep, n, chi, M):
    """Ragone Eq.10 prediction Var_pred = P_g(rho) P_g(O)/dim(g).

    For cqfa/classical_fa use the RESTRICTED (frame-induced) DLA on H_+ and the
    projected chi; for non_equivariant use the AMBIENT su(2^n) DLA and the raw
    state.  Returns (var_pred, dim_g) or (nan, nan) if not computed.
    """
    try:
        if model_name in ("cqfa", "classical_fa"):
            d_eff, basis, Bplus = D.frame_induced_dla(rep, n)
            if d_eff == 0:
                return float("nan"), float("nan")
            chi_r = Bplus.conj().T @ chi
            nrm = np.linalg.norm(chi_r)
            if nrm < 1e-12:
                return float("nan"), float(d_eff)
            chi_r = chi_r / nrm
            rho_r = np.outer(chi_r, chi_r.conj())
            M_r = D.restrict(M, Bplus)
            return D.ragone_variance(rho_r, M_r, basis, d_eff), float(d_eff)
        else:  # non_equivariant: ambient DLA (capped for speed)
            gens = D.hea_circuit_generators(n)
            d_amb, basis = D.lie_closure_dim(gens, cap=4 ** n - 1)
            rho = np.outer(chi, chi.conj())
            return D.ragone_variance(rho, M, basis, d_amb), float(d_amb)
    except Exception:
        return float("nan"), float("nan")


def sweep(args):
    ns = list(range(args.n_min, args.n_max + 1))
    mu_kinds = ["early", "mid", "max"]
    var_rows = []
    # per (model, mu_kind): lists of across-seed mean variances (one per n) and dims
    curves = {m: {mk: {"mean": [], "dim": []} for mk in mu_kinds} for m in args.models}

    for n in ns:
        rep = G.z2_pauli(n)
        mu_dict = A.component_indices(n, args.n_layers)
        # invariant observable with P_g(M) > 0 (V4); reused by all models for fairness
        M, ok = reachable_observable(rep, n)
        for model_name in args.models:
            model = _build_model(model_name, rep, args.n_layers, M)
            t0 = time.time()
            # per outer seed: variance at each mu-kind + pooled raw values for CI
            seed_var = {mk: [] for mk in mu_kinds}
            pooled_raw = {mk: [] for mk in mu_kinds}
            p_norms = []
            var_pred_val = float("nan")
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                # fixed input(s)
                inputs = [A.fixed_input_for(model, rep, rng)
                          for _ in range(args.n_inputs)]
                # variance is averaged over the K fixed inputs
                per_input = {mk: [] for mk in mu_kinds}
                for psi in inputs:
                    phi = rep.projector() @ psi
                    p = float(np.real(phi.conj() @ phi))
                    p_norms.append(p)
                    res = A.gradient_variance_all(model, psi, args.samples, rng, mu_dict)
                    for mk in mu_kinds:
                        per_input[mk].append(res[mk][0])
                        pooled_raw[mk].extend(res[mk][1].tolist())
                for mk in mu_kinds:
                    seed_var[mk].append(float(np.mean(per_input[mk])))
                # DLA prediction (computed once, on the last seed's first input)
                if np.isnan(var_pred_val):
                    chi = phi / np.sqrt(p) if p > 1e-12 else phi
                    var_pred_val, _ = _dla_prediction(model_name, rep, n, chi, M)
            wall = time.time() - t0

            for mk in mu_kinds:
                sv = np.array(seed_var[mk])
                lo, hi = A.bootstrap_var_ci(np.array(pooled_raw[mk]),
                                            n_boot=args.boot, seed=0)
                var_rows.append(dict(
                    model=model_name, n=n, dim=2 ** n, n_layers=args.n_layers,
                    n_params=model.n_params, mu_kind=mk, mu_index=(
                        mu_dict.get(mk, -1) if mk != "max" else -1),
                    samples=args.samples, seeds=len(args.seeds),
                    var_mean=float(np.mean(sv)), var_std=float(np.std(sv)),
                    var_ci_lo=lo, var_ci_hi=hi,
                    var_pred=var_pred_val,
                    p_norm_mean=(float(np.mean(p_norms))
                                 if model_name == "cqfa" else float("nan")),
                    wall_s=round(wall, 3)))
                curves[model_name][mk]["mean"].append(float(np.mean(sv)))
                # dim for cross-check (recompute cheaply per point)
                if mk == "early":
                    _, dim_g = _dla_prediction(
                        model_name, rep, n,
                        (rep.projector() @ A.random_pure_state(
                            2 ** n, np.random.default_rng(0))), M)
                    for mk2 in mu_kinds:
                        curves[model_name][mk2]["dim"].append(dim_g)
            print(f"  n={n:>2} {model_name:<16} "
                  f"V_early={curves[model_name]['early']['mean'][-1]:.3e} "
                  f"(pred {var_pred_val:.3e}) {wall:.1f}s")

    return ns, var_rows, curves, mu_kinds


def fit_all(ns, curves, mu_kinds, models, boot):
    fit_rows = []
    for model_name in models:
        for mk in mu_kinds:
            V = np.array(curves[model_name][mk]["mean"])
            dims = np.array(curves[model_name][mk]["dim"])
            npts = len(V)
            b, _, r2_exp = A.fit_exp(ns, V)
            c, _, r2_poly = A.fit_poly(ns, V)
            b_lo, b_hi = A.bootstrap_slope_ci(
                ns, np.log(np.clip(V, 1e-300, None)),
                lambda x, y: A._ols(x, y), n_boot=boot, seed=1)
            c_lo, c_hi = A.bootstrap_slope_ci(
                np.log(ns), np.log(np.clip(V, 1e-300, None)),
                lambda x, y: A._ols(x, y), n_boot=boot, seed=2)
            cross_slope, r2_cross = A.crosscheck_fit(V, dims)
            cls = A.classify_decay(b, r2_exp, r2_poly)
            fit_rows.append(dict(
                model=model_name, n_points=npts, mu_kind=mk,
                exp_slope_b=b, exp_slope_b_ci_lo=-b_hi, exp_slope_b_ci_hi=-b_lo,
                r2_exp=r2_exp, poly_exp_c=c, poly_exp_c_ci_lo=-c_hi,
                poly_exp_c_ci_hi=-c_lo, r2_poly=r2_poly,
                crosscheck_slope=cross_slope, r2_crosscheck=r2_cross,
                classification=cls))
    return fit_rows


def main():
    ap = argparse.ArgumentParser(description="Gradient-variance sweep")
    ap.add_argument("--n-min", type=int, default=2)
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-inputs", type=int, default=1)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--models", type=str, nargs="+",
                    default=["cqfa", "non_equivariant", "classical_fa"])
    ap.add_argument("--out", type=str,
                    default="experiments/trainability_law/out/variance.csv")
    ap.add_argument("--fits-out", type=str,
                    default="experiments/trainability_law/out/fits.csv")
    args = ap.parse_args()

    print(f"[scaling] numpy {np.__version__}  n={args.n_min}..{args.n_max}  "
          f"S={args.samples}  seeds={args.seeds}  layers={args.n_layers}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    ns, var_rows, curves, mu_kinds = sweep(args)
    fit_rows = fit_all(ns, curves, mu_kinds, args.models, args.boot)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=VAR_FIELDS)
        w.writeheader()
        for r in var_rows:
            w.writerow(r)
    with open(args.fits_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIT_FIELDS)
        w.writeheader()
        for r in fit_rows:
            w.writerow(r)

    manifest = {"script": "run_scaling", "numpy": np.__version__,
                "args": vars(args), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "n_var_rows": len(var_rows), "n_fit_rows": len(fit_rows)}
    with open(os.path.splitext(args.out)[0] + "_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[scaling] wrote {args.out} ({len(var_rows)} rows) and "
          f"{args.fits_out} ({len(fit_rows)} rows)")
    for r in fit_rows:
        if r["mu_kind"] == "early":
            print(f"  [{r['model']:<16}] b={r['exp_slope_b']:.3f} R2_exp={r['r2_exp']:.3f} "
                  f"c={r['poly_exp_c']:.3f} R2_poly={r['r2_poly']:.3f} -> {r['classification']}")


if __name__ == "__main__":
    main()
