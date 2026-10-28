
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

warnings.filterwarnings("ignore")

# single-threaded BLAS per worker so N workers == N cores
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from .dlp_real import (_primitive_root, dlog_table, concept, classical_features,
                       quantum_feature, _classical_models, _cv_accuracy)

# Large-prime ladder, roughly geometric above the existing p<=4001 sweep.
# Staged because per-unit cost grows ~n^1.2 (trees/MLP) with an O(n^2) SVC term.
PRIMES_STAGE1 = [5003, 6007, 8009, 10007, 12007, 14009, 16007]
PRIMES_STAGE2 = [20011, 24007, 28001, 32003]
PRIMES_STAGE3 = [36007, 40009, 45007, 50021]

OLD_AGG = "results/discrete_log/dlp_wide/dlp_wide_agg.csv"


def run_unit(p, seed, out_dir):
    """One (prime, seed) unit -- byte-for-byte the same computation as
    dlp_wide.run_unit, just a different parts directory."""
    part = os.path.join(out_dir, "parts", f"p{p}_s{seed}.json")
    if os.path.exists(part):
        return part, False, 0.0                          # resume: already done
    t0 = time.time()
    g = _primitive_root(p)
    dl = dlog_table(p, g)
    rng = np.random.default_rng(seed)
    s = int(rng.integers(0, p - 1))
    xs, y = concept(p, g, s, dl)
    Xc = classical_features(xs, p)
    row = {"prime": p, "seed": seed, "generator": g, "n_examples": p - 1}
    for name, model in _classical_models().items():
        row[f"cls_{name}"] = _cv_accuracy(model, Xc, y, seed=seed)
    from sklearn.linear_model import LogisticRegression
    row["cls_rawvalue_only"] = _cv_accuracy(
        LogisticRegression(C=10.0, max_iter=2000), (xs / p).reshape(-1, 1), y, seed=seed)
    Xq = quantum_feature(xs, dl, p)
    row["quantum_dlog"] = _cv_accuracy(
        LogisticRegression(C=10.0, max_iter=2000), Xq, y, seed=seed)
    row["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(part), exist_ok=True)
    with open(part, "w") as f:
        json.dump(row, f)
    return part, True, row["seconds"]


def _aggregate(rows):
    """Per-prime aggregate (mean over seeds), same reduction as dlp_wide.merge."""
    agg = {}
    for r in rows:
        agg.setdefault(r["prime"], []).append(r)
    cls_keys = [k for k in rows[0] if k.startswith("cls_")]
    arows = []
    for p, rs in sorted(agg.items()):
        best_cls = max(float(np.mean([r[k] for r in rs])) for k in cls_keys)
        arows.append({
            "prime": p,
            "n_seeds": len(rs),
            "quantum_acc": round(float(np.mean([r["quantum_dlog"] for r in rs])), 4),
            "quantum_std": round(float(np.std([r["quantum_dlog"] for r in rs])), 4),
            "best_classical_acc": round(best_cls, 4),
            **{k + "_mean": round(float(np.mean([r[k] for r in rs])), 4) for k in cls_keys},
            "source": "dlp_wide_large",
        })
    return arows


def _carry_over_old():
    """Read the existing p<=4001 aggregate and re-express it in the new column
    names. Values are COPIED verbatim -- nothing is recomputed or invented."""
    if not os.path.exists(OLD_AGG):
        return []
    out = []
    with open(OLD_AGG) as f:
        for r in csv.DictReader(f):
            row = {
                "prime": int(r["prime"]),
                "n_seeds": int(r["n_seeds"]),
                "quantum_acc": float(r["quantum_dlog_mean"]),
                "quantum_std": float(r["quantum_dlog_std"]),
                "best_classical_acc": float(r["best_classical_mean"]),
            }
            for k, v in r.items():
                if k.startswith("cls_"):
                    row[k] = float(v)
            row["source"] = "dlp_wide"
            out.append(row)
    return out


def merge(out_dir):
    parts_dir = os.path.join(out_dir, "parts")
    if not os.path.isdir(parts_dir):
        return None
    rows = []
    for fn in sorted(os.listdir(parts_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(parts_dir, fn)) as f:
                rows.append(json.load(f))
    if not rows:
        return None
    rows.sort(key=lambda r: (r["prime"], r["seed"]))
    raw = os.path.join(out_dir, "dlp_wide_large.csv")
    with open(raw, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    new = _aggregate(rows)
    full = _carry_over_old() + new
    full.sort(key=lambda r: r["prime"])
    fields = ["prime", "n_seeds", "quantum_acc", "quantum_std", "best_classical_acc"]
    for r in full:
        for k in r:
            if k not in fields and k != "source":
                fields.append(k)
    fields.append("source")
    aggp = "results/discrete_log/dlp_wide_agg_large.csv"
    with open(aggp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader(); w.writerows(full)
    return raw, aggp, len(rows), new


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--primes", type=int, nargs="+", default=None)
    ap.add_argument("--stage", type=int, default=0, choices=[0, 1, 2, 3])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4)))
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--out", default="results/discrete_log/dlp_wide_large")
    args = ap.parse_args(argv)

    if not args.merge_only:
        primes = args.primes or {1: PRIMES_STAGE1, 2: PRIMES_STAGE2,
                                 3: PRIMES_STAGE3, 0: PRIMES_STAGE1}[args.stage]
        units = [(p, s) for p in primes for s in range(args.seeds)]
        os.makedirs(os.path.join(args.out, "parts"), exist_ok=True)
        done = ran = 0
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_unit, p, s, args.out): (p, s) for p, s in units}
            for fut in as_completed(futs):
                _, fresh, secs = fut.result()
                p, s = futs[fut]
                ran += int(fresh); done += 1
                print(f"[large] {done}/{len(units)} p={p} seed={s} "
                      f"{'fresh %.0fs' % secs if fresh else 'cached'} "
                      f"| elapsed {time.time()-t0:.0f}s", flush=True)

    out = merge(args.out)
    if out:
        raw, aggp, n, new = out
        print(f"\n[large] merged {n} unit rows -> {raw}")
        print(f"[large] full curve -> {aggp}")
        print(f"{'prime':>7} {'seeds':>5} {'quantum':>8} {'best_cls':>9}")
        for r in new:
            print(f"{r['prime']:>7} {r['n_seeds']:>5} {r['quantum_acc']:>8.4f} "
                  f"{r['best_classical_acc']:>9.4f}")


if __name__ == "__main__":
    main()
