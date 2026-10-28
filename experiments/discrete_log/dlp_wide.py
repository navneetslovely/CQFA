"""
experiments/discrete_log/dlp_wide.py
========================================
The wide discrete-log sweep: turns the 3-prime / 5-seed head-to-head into a
CURVE -- many primes x many seeds x all six classical models, with a raw-value
feature ablation. Addresses the review finding "thin, small-scale evidence: only
3 primes, ~50 examples".

Resume-aware CPU-fleet driver: each unit = (prime, seed) runs the full model bank
and writes results/discrete_log/dlp_wide/parts/p{p}_s{seed}.json ; finished
units are SKIPPED on relaunch, so the job can be killed/restarted freely (server
rule: relaunching continues, never restarts). Final merge -> dlp_wide.csv .

Run (Mac quick):   python -m experiments.discrete_log.dlp_wide --quick
Run (server):      python -m experiments.discrete_log.dlp_wide --jobs 28
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

warnings.filterwarnings("ignore")

# single-threaded BLAS per worker so N workers == N cores (server rule)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from .dlp_real import (_primitive_root, dlog_table, concept, classical_features,
                       quantum_feature, _classical_models, _cv_accuracy)


PRIMES_FULL = [23, 31, 47, 59, 83, 101, 127, 151, 179, 199, 251, 293, 331, 383, 433, 499]
PRIMES_QUICK = [23, 47, 83]


def run_unit(p, seed, out_dir):
    """One (prime, seed) unit: all 6 classical models on rich raw-x features, a
    raw-value-only ablation (logistic on x/p alone), and the quantum dlog feature."""
    part = os.path.join(out_dir, "parts", f"p{p}_s{seed}.json")
    if os.path.exists(part):
        return part, False                               # resume: already done
    g = _primitive_root(p)
    dl = dlog_table(p, g)
    rng = np.random.default_rng(seed)
    s = int(rng.integers(0, p - 1))
    xs, y = concept(p, g, s, dl)
    Xc = classical_features(xs, p)
    row = {"prime": p, "seed": seed, "generator": g, "n_examples": p - 1}
    for name, model in _classical_models().items():
        row[f"cls_{name}"] = _cv_accuracy(model, Xc, y, seed=seed)
    # ablation: raw normalised value only (no engineered periodic bank)
    from sklearn.linear_model import LogisticRegression
    row["cls_rawvalue_only"] = _cv_accuracy(
        LogisticRegression(C=10.0, max_iter=2000), (xs / p).reshape(-1, 1), y, seed=seed)
    Xq = quantum_feature(xs, dl, p)
    row["quantum_dlog"] = _cv_accuracy(
        LogisticRegression(C=10.0, max_iter=2000), Xq, y, seed=seed)
    os.makedirs(os.path.dirname(part), exist_ok=True)
    with open(part, "w") as f:
        json.dump(row, f)
    return part, True


def merge(out_dir):
    parts_dir = os.path.join(out_dir, "parts")
    rows = []
    for fn in sorted(os.listdir(parts_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(parts_dir, fn)) as f:
                rows.append(json.load(f))
    rows.sort(key=lambda r: (r["prime"], r["seed"]))
    if not rows:
        return None
    raw = os.path.join(out_dir, "dlp_wide.csv")
    with open(raw, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # per-prime aggregate (mean over seeds): quantum vs best-classical curve
    agg = {}
    for r in rows:
        agg.setdefault(r["prime"], []).append(r)
    cls_keys = [k for k in rows[0] if k.startswith("cls_")]
    arows = []
    for p, rs in sorted(agg.items()):
        best_cls = max(float(np.mean([r[k] for r in rs])) for k in cls_keys)
        arows.append({
            "prime": p, "n_seeds": len(rs),
            "quantum_dlog_mean": round(float(np.mean([r["quantum_dlog"] for r in rs])), 4),
            "quantum_dlog_std": round(float(np.std([r["quantum_dlog"] for r in rs])), 4),
            "best_classical_mean": round(best_cls, 4),
            **{k + "_mean": round(float(np.mean([r[k] for r in rs])), 4) for k in cls_keys},
        })
    aggp = os.path.join(out_dir, "dlp_wide_agg.csv")
    with open(aggp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(arows[0].keys())); w.writeheader(); w.writerows(arows)
    return raw, aggp, len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small Mac sanity config")
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--primes", type=int, nargs="+", default=None, help="override prime list")
    ap.add_argument("--out", default="results/discrete_log/dlp_wide")
    args = ap.parse_args(argv)
    primes = args.primes if args.primes else (PRIMES_QUICK if args.quick else PRIMES_FULL)
    seeds = range(3 if args.quick else args.seeds)
    units = [(p, s) for p in primes for s in seeds]
    os.makedirs(os.path.join(args.out, "parts"), exist_ok=True)
    done = ran = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_unit, p, s, args.out): (p, s) for p, s in units}
        for fut in as_completed(futs):
            _, fresh = fut.result()
            ran += int(fresh); done += 1
            if done % 25 == 0 or done == len(units):
                print(f"[dlp_wide] {done}/{len(units)} units ({ran} fresh)", flush=True)
    out = merge(args.out)
    if out:
        raw, aggp, n = out
        print(f"[dlp_wide] merged {n} rows -> {raw} ; per-prime curve -> {aggp}")


if __name__ == "__main__":
    main()
