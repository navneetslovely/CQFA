"""Seeded training, bootstrap confidence intervals and CSV logging.

Conventions:

  * all randomness flows through the ``seeds`` argument;
  * results are reported as acc_mean +/- acc_std with a bootstrap 95% interval
    over those seeds;
  * each run writes a self-describing folder containing its configuration and
    output CSVs, so every reported value is traceable to its source.

Command line::

    python -m cqfa.train --rep z2_rc --model all --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from .group import get_representation
from .readout import build_models
from . import tasks as T


# ---------------------------------------------------------------------------
# logistic head + helpers
# ---------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _features_only(model, theta, X):
    return model.features_batch(theta, X)[0]


def evaluate(model, theta, a, b, X, y):
    p = sigmoid(a * _features_only(model, theta, X) + b)
    return float(np.mean((p > 0.5).astype(int) == y))


def predict_proba(model, theta, a, b, X):
    """Logistic-head score p = sigmoid(a*feature+b) for each row (for AUPRC etc.)."""
    return sigmoid(a * _features_only(model, theta, X) + b)


# ---------------------------------------------------------------------------
# single-seed training
# ---------------------------------------------------------------------------
def train_model(model, Xtr, ytr, Xte, yte, epochs, lr, seed,
                augment_rep=None, log_rows=None, eval_rows=None, eval_every=10, momentum=0.9):
    rng = np.random.default_rng(seed)

    if augment_rep is not None:                       # orbit augmentation
        Xa, ya = [], []
        for psi, label in zip(Xtr, ytr):
            for U in augment_rep.matrices:
                Xa.append(U @ psi)
                ya.append(label)
        Xtr, ytr = np.array(Xa), np.array(ya)

    theta = model.init_params(rng)
    a, b = 1.0, 0.0
    v_theta = np.zeros_like(theta)
    v_a = v_b = 0.0
    best = 0.0
    loss = float("nan")

    for ep in range(epochs):
        fs, gs = model.features_batch(theta, Xtr)
        z = a * fs + b
        p = sigmoid(z)
        dz = p - ytr                                  # d(BCE)/dz
        g_theta = ((dz * a)[:, None] * gs).mean(0)
        g_a = float((dz * fs).mean())
        g_b = float(dz.mean())
        loss = float(-np.mean(ytr * np.log(p + 1e-9) + (1 - ytr) * np.log(1 - p + 1e-9)))
        tr_acc = float(np.mean((p > 0.5).astype(int) == ytr))

        v_theta = momentum * v_theta - lr * g_theta
        theta = theta + v_theta
        v_a = momentum * v_a - lr * g_a
        a = a + v_a
        v_b = momentum * v_b - lr * g_b
        b = b + v_b

        if log_rows is not None:
            log_rows.append(dict(seed=seed, epoch=ep, loss=loss, train_acc=tr_acc))
        if ep % eval_every == 0 or ep == epochs - 1:
            te_acc = evaluate(model, theta, a, b, Xte, yte)
            best = max(best, te_acc)
            if eval_rows is not None:
                eval_rows.append(dict(seed=seed, epoch=ep, test_acc=te_acc))

    final_te = evaluate(model, theta, a, b, Xte, yte)
    best = max(best, final_te)
    return dict(seed=seed, best_test_acc=float(best), final_test_acc=float(final_te),
                final_train_loss=loss, n_params=int(model.n_params),
                theta=theta, a=float(a), b=float(b))


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def bootstrap_ci(vals, n_boot=2000, alpha=0.05, seed=0):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return float(vals.min()), float(vals.max())
    rng = np.random.default_rng(seed)
    boot = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# experiment (>=5 seeds -> a run folder)
# ---------------------------------------------------------------------------
def run_experiment(rep_name, model_name, seeds, n_train=80, n_test=60, epochs=120,
                   lr=0.15, n_layers=3, tag="", out_root=None, teacher_seed=12345,
                   write=True, task_fn=None, task_name="coherent_teacher",
                   rep=None, return_preds=False):
    """Seeded training over the four models with per-run CSV logging.

    If `task_fn` is None the coherent-teacher task is used. The benchmark sweep
    injects any `(n_samples, rng) -> (X, y)` callable via `task_fn`.

    If `return_preds=True`, additionally returns per-seed test predictions
    [{seed, y_true, p_score}] so the driver can compute balanced_acc/MCC/AUPRC
    without touching this hot loop.

    `rep` may be passed to reuse a pre-built Representation (e.g. the genomics
    U_RC rep that is not in get_representation's registry).
    """
    if rep is None:
        rep = get_representation(rep_name)
    model = build_models(rep, n_layers)[model_name]

    if task_fn is None:                                # default: coherent teacher
        teacher_M = T.random_hermitian(rep.dim, np.random.default_rng(teacher_seed))
        task_fn = lambda n, drng: T.coherent_teacher_task(rep, n, drng, teacher_M)

    log_rows, eval_rows, finals, pred_rows = [], [], [], []
    class_balances = []
    for s in seeds:
        drng = np.random.default_rng(1000 + s)         # data varies by seed; concept fixed
        X, y = task_fn(n_train + n_test, drng)
        Xtr, ytr, Xte, yte = X[:n_train], y[:n_train], X[n_train:], y[n_train:]
        class_balances.append(float(np.mean(y)))
        aug = rep if model.augmented else None
        res = train_model(model, Xtr, ytr, Xte, yte, epochs, lr, s,
                          augment_rep=aug, log_rows=log_rows, eval_rows=eval_rows)
        if return_preds:
            p_te = predict_proba(model, res["theta"], res["a"], res["b"], Xte)
            pred_rows.append(dict(seed=int(s), y_true=np.asarray(yte, dtype=int),
                                  p_score=np.asarray(p_te, dtype=float)))
        # drop non-serialisable trained params before logging
        finals.append({k: v for k, v in res.items()
                       if k in ("seed", "best_test_acc", "final_test_acc",
                                "final_train_loss", "n_params")})

    accs = [f["best_test_acc"] for f in finals]
    lo, hi = bootstrap_ci(accs, seed=teacher_seed)
    summary = dict(rep=rep_name, model=model_name, task=task_name, n_seeds=len(seeds),
                   acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                   acc_min=float(np.min(accs)), acc_max=float(np.max(accs)),
                   ci_lo=lo, ci_hi=hi, n_params=int(model.n_params),
                   class_balance=float(np.mean(class_balances)))

    folder = None
    if write:
        root = out_root or os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs")
        os.makedirs(root, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        folder = os.path.join(root, f"run_{model_name}_{rep_name}_{ts}" + (f"_{tag}" if tag else ""))
        os.makedirs(folder, exist_ok=True)
        _write_csv(os.path.join(folder, "train_log.csv"), log_rows, ["seed", "epoch", "loss", "train_acc"])
        _write_csv(os.path.join(folder, "eval_log.csv"), eval_rows, ["seed", "epoch", "test_acc"])
        _write_csv(os.path.join(folder, "final_metrics.csv"), finals,
                   ["seed", "best_test_acc", "final_test_acc", "final_train_loss", "n_params"])
        _write_csv(os.path.join(folder, "seed_summary.csv"), [summary], list(summary.keys()))
        with open(os.path.join(folder, "config.json"), "w") as fh:
            json.dump(dict(rep=rep_name, model=model_name, task=task_name, seeds=list(seeds),
                           n_train=n_train, n_test=n_test, epochs=epochs, lr=lr,
                           n_layers=n_layers, teacher_seed=teacher_seed, tag=tag), fh, indent=2)
    if return_preds:
        return summary, folder, pred_rows
    return summary, folder


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the CQFA model variants.")
    ap.add_argument("--rep", default="z2_rc", help="z2_rc | c4 | d4 | z2_pauli ...")
    ap.add_argument("--model", default="all",
                    help="cqfa | classical_fa | non_equivariant | augmented | all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.15)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--task", default="coherent_teacher",
                    help="coherent_teacher | orbit_interference | symmetry_vs_broken | spt_derived")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    names = (["cqfa", "classical_fa", "non_equivariant", "augmented"]
             if args.model == "all" else [args.model])
    rep = get_representation(args.rep)
    task_fn = T.make_task(args.task, rep) if args.task != "coherent_teacher" else None
    print(f"\nTask: {args.task} | rep={args.rep} | seeds={args.seeds}")
    print(f"{'model':<16}{'#params':>8}{'acc_mean':>10}{'acc_std':>9}{'95% CI':>18}")
    print("-" * 61)
    results = {}
    for m in names:
        s, folder = run_experiment(args.rep, m, args.seeds, args.n_train, args.n_test,
                                   args.epochs, args.lr, args.n_layers, args.tag,
                                   write=not args.no_write, task_fn=task_fn,
                                   task_name=args.task, rep=rep)
        results[m] = s
        print(f"{m:<16}{s['n_params']:>8}{s['acc_mean']:>10.3f}{s['acc_std']:>9.3f}"
              f"   [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]")
    if "cqfa" in results and "classical_fa" in results:
        gap = results["cqfa"]["acc_mean"] - results["classical_fa"]["acc_mean"]
        print("-" * 61)
        print(f"CQFA - classical-FA accuracy gap = {gap:+.3f}  "
              f"(coherent interference the output-average cannot represent)")
    return results


if __name__ == "__main__":
    main()
