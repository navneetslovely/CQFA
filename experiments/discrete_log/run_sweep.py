
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime

import numpy as np

from cqfa.train import bootstrap_ci
from .dlog_concept import DlogConcept, make_dataset, random_secret
from .learners import (
    fiducial_state,
    structured_accuracy,
    classical_raw_accuracy,
    chance_accuracy,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def run(p: int, seeds: int, n_train: int, n_test: int, C: float = 1.0,
        base_seed: int = 0, tag: str = "", out_root: str | None = None,
        write: bool = True, verbose: bool = True):
    """Run the sweep over `seeds` seeds at prime `p`. Returns (summary_dict, folder)."""
    concept = DlogConcept.build(p)
    # a single fixed fiducial state for the covariant kernel (shared across seeds,
    # deterministic) -- covariance means the kernel only depends on log differences
    phi0 = fiducial_state(concept.order)

    per_seed = []
    for i in range(seeds):
        seed = base_seed + i
        rng = np.random.default_rng(seed)
        s = random_secret(concept, rng)
        x_tr, y_tr, x_te, y_te = make_dataset(concept, s, n_train, n_test, rng)

        acc_struct = structured_accuracy(concept, x_tr, y_tr, x_te, y_te, phi0=phi0, C=C)
        acc_class = classical_raw_accuracy(concept, x_tr, y_tr, x_te, y_te)
        acc_chance = chance_accuracy(y_te)
        per_seed.append(dict(seed=seed, secret=s,
                             structured_acc=acc_struct,
                             classical_raw_acc=acc_class,
                             chance_acc=acc_chance))
        if verbose:
            print(f"  seed={seed:>3} s={s:>3}  structured={acc_struct:.3f}  "
                  f"classical_raw={acc_class:.3f}  chance={acc_chance:.3f}")

    def col(key):
        return np.array([r[key] for r in per_seed], dtype=float)

    struct, classical, chance = col("structured_acc"), col("classical_raw_acc"), col("chance_acc")
    s_lo, s_hi = bootstrap_ci(struct, seed=base_seed)
    c_lo, c_hi = bootstrap_ci(classical, seed=base_seed)

    summary = dict(
        p=p, generator=concept.g, group_order=concept.order,
        n_train=n_train, n_test=n_test, seeds=seeds, C=C,
        structured_mean=float(struct.mean()), structured_std=float(struct.std(ddof=0)),
        structured_ci_lo=s_lo, structured_ci_hi=s_hi,
        classical_raw_mean=float(classical.mean()), classical_raw_std=float(classical.std(ddof=0)),
        classical_raw_ci_lo=c_lo, classical_raw_ci_hi=c_hi,
        chance_mean=float(chance.mean()),
        separation_mean=float(struct.mean() - classical.mean()),
    )

    folder = None
    if write:
        out_root = out_root or os.path.join(_REPO_ROOT, "runs")
        os.makedirs(out_root, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = f"_{tag}" if tag else ""
        folder = os.path.join(out_root, f"run_dlog_p{p}_{stamp}{suffix}")
        os.makedirs(folder, exist_ok=True)
        _write_csv(os.path.join(folder, "per_seed.csv"), per_seed,
                   ["seed", "secret", "structured_acc", "classical_raw_acc", "chance_acc"])
        _write_csv(os.path.join(folder, "summary.csv"), [summary], list(summary.keys()))

    if verbose:
        print("-" * 68)
        print(f"  STRUCTURED (covariant C_{{{concept.order}}}, log-aware): "
              f"{summary['structured_mean']:.3f} +/- {summary['structured_std']:.3f}  "
              f"95% CI [{s_lo:.3f}, {s_hi:.3f}]")
        print(f"  CLASSICAL  (raw-x, log-blind, best-of-family):     "
              f"{summary['classical_raw_mean']:.3f} +/- {summary['classical_raw_std']:.3f}  "
              f"95% CI [{c_lo:.3f}, {c_hi:.3f}]")
        print(f"  chance (majority class):                           {summary['chance_mean']:.3f}")
        print(f"  SEPARATION (structured - classical):               {summary['separation_mean']:+.3f}")
        if folder:
            print(f"  CSVs -> {folder}")
    return summary, folder


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def main():
    ap = argparse.ArgumentParser(description="Discrete-log concept-class proof-of-principle")
    ap.add_argument("--p", type=int, default=23, help="prime (small; simulable)")
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (>=5 for headline)")
    ap.add_argument("--n_train", type=int, default=8, help="train samples (few-shot)")
    ap.add_argument("--n_test", type=int, default=14, help="disjoint test samples")
    ap.add_argument("--C", type=float, default=1.0, help="SVM regularisation")
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--no_write", action="store_true", help="do not write CSVs")
    args = ap.parse_args()

    print(f"Discrete-log concept class  p={args.p}  seeds={args.seeds}  "
          f"n_train={args.n_train}  n_test={args.n_test}")
    print("  (proof-of-principle at simulable p; NO scaling claim; classical")
    print("   hardness = LAT/DLP theorem, CITED not re-proven)")
    run(args.p, args.seeds, args.n_train, args.n_test, C=args.C,
        base_seed=args.base_seed, tag=args.tag, write=not args.no_write)


if __name__ == "__main__":
    main()
