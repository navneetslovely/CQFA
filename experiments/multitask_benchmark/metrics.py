"""
experiments.multitask_benchmark.metrics
=============================
Post-hoc classification metrics for the benchmark sweep, computed from per-seed
test predictions (y_true, p_score) OUTSIDE the training hot loop.

Metrics: accuracy, balanced_accuracy, MCC, AUPRC. Each is aggregated over seeds
as mean +/- std with a bootstrap 95% CI (reusing cqfa.train.bootstrap_ci for a
single, consistent CI implementation across the paper).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             matthews_corrcoef, average_precision_score)

from cqfa.train import bootstrap_ci


def per_seed_metrics(y_true, p_score):
    """Four metrics for one seed's test predictions.

    y_true : int array in {0,1}; p_score : float array (logistic score in [0,1]).
    MCC/balanced_acc are robust to class imbalance; AUPRC is the imbalanced-case
    score. AUPRC/MCC guard against degenerate single-class test folds.
    """
    y_true = np.asarray(y_true, dtype=int)
    p_score = np.asarray(p_score, dtype=float)
    y_pred = (p_score > 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    balacc = float(balanced_accuracy_score(y_true, y_pred))
    # MCC is 0 when a prediction/truth vector is constant (undefined denominator).
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        mcc = 0.0
    else:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    # AUPRC needs both classes present in y_true.
    if len(np.unique(y_true)) < 2:
        auprc = float(np.mean(y_true))          # trivial baseline = positive rate
    else:
        auprc = float(average_precision_score(y_true, p_score))
    return dict(accuracy=acc, balanced_accuracy=balacc, mcc=mcc, auprc=auprc)


def aggregate(pred_rows, ci_seed: int = 12345):
    """Aggregate per-seed predictions into mean/std/CI for each of the 4 metrics.

    pred_rows : list of dict(seed, y_true, p_score).
    Returns dict with <metric>_mean/_std/_ci_lo/_ci_hi for
    metric in {acc, balacc, mcc, auprc}.
    """
    keys = ["accuracy", "balanced_accuracy", "mcc", "auprc"]
    short = {"accuracy": "acc", "balanced_accuracy": "balacc",
             "mcc": "mcc", "auprc": "auprc"}
    per = [per_seed_metrics(r["y_true"], r["p_score"]) for r in pred_rows]
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in per], dtype=float)
        lo, hi = bootstrap_ci(vals, seed=ci_seed)
        s = short[k]
        out[f"{s}_mean"] = float(np.mean(vals))
        out[f"{s}_std"] = float(np.std(vals))
        out[f"{s}_ci_lo"] = lo
        out[f"{s}_ci_hi"] = hi
    return out
