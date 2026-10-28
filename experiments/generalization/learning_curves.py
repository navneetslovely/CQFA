
from __future__ import annotations
import argparse, csv, os
import numpy as np

from cqfa.train import run_experiment


def run(rep, models, train_sizes, seeds, epochs, n_test, out_dir):
    rows = []
    for m in models:
        for nt in train_sizes:
            s, _ = run_experiment(rep, m, seeds, n_train=nt, n_test=n_test,
                                  epochs=epochs, tag="gen", write=False)
            rows.append(dict(rep=rep, model=m, n_train=nt, n_seeds=len(seeds),
                             acc_mean=s["acc_mean"], acc_std=s["acc_std"],
                             ci_lo=s["ci_lo"], ci_hi=s["ci_hi"], n_params=s["n_params"]))
            print(f"{m:<16} n_train={nt:>4}  acc={s['acc_mean']:.3f} +/- {s['acc_std']:.3f}")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "learning_curves.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", p)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="z2_rc")
    ap.add_argument("--models", nargs="+", default=["cqfa", "classical_fa", "non_equivariant"])
    ap.add_argument("--train-sizes", type=int, nargs="+", default=[8, 16, 24, 40, 64, 100, 160])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--out", default="results/generalization")
    a = ap.parse_args(argv)
    run(a.rep, a.models, a.train_sizes, a.seeds, a.epochs, a.n_test, a.out)


if __name__ == "__main__":
    main()
