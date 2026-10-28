"""
experiments/discrete_log/dlp_baselines_tuned.py
===================================================
Properly TUNED classical baselines on the discrete-log concept.

The head-to-head in dlp_real.py / dlp_wide.py reports the best of six classical
models at standard configurations, which leaves the baselines undertuned. Here
every classical family gets a full cross-validated hyperparameter search on the
same data, splits and seeds, plus a small neural network trained with Adam.

Protocol (identical data pipeline to dlp_real.py):
  * for prime p, primitive root g, seed sd: s = rng(sd).integers(0, p-1);
    x = 1..p-1, y = 1[ (dlog_g(x) - s) mod (p-1) < (p-1)/2 ].
  * classical learners see x only, through a feature map:
      - "rich16"       : the existing bank (x/p, (x/p)^2, 16 sin/cos harmonics)
      - "full_fourier" : the COMPLETE Fourier basis in x-space
                         (all k = 1..min((p-1)//2, HARM_CAP) sin/cos harmonics)
  * quantum reference: the dlog feature map (cos, sin of 2*pi*dlog/(p-1)) +
    plain logistic regression -- untuned, exactly as in dlp_real.py.

Two accuracies are reported per model, both 5-fold StratifiedKFold (same splitter
as _cv_accuracy in dlp_real.py, scaler inside the pipeline):
  * cv_acc_tuned    -- the search's own best inner-CV score. This is optimistically
                       biased, since the configuration was chosen on this very
                       score, and is therefore the most generous number available
                       to the classical side. It is reported deliberately.
  * nested_cv_acc   -- nested CV, with the grid re-selected inside each outer fold:
                       the selection-bias-free generalisation estimate.
Both are reported.

Output: results/discrete_log/dlp_baselines_tuned.csv
"""
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
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from .dlp_real import _primitive_root, dlog_table, concept, quantum_feature

HARM_CAP = 128          # cap on #harmonics for the "complete" Fourier basis
CV_FOLDS = 5


# ---------------------------------------------------------------------------
# feature maps
# ---------------------------------------------------------------------------
def features(xs, p, kind):
    x = xs.astype(float)
    if kind == "rich16":
        n_harm = 16
        feats = [x / p, (x / p) ** 2,
                 np.sin(2 * np.pi * x / p), np.cos(2 * np.pi * x / p)]
    elif kind == "full_fourier":
        n_harm = min((p - 1) // 2, HARM_CAP)
        feats = [x / p]
    else:
        raise ValueError(kind)
    for k in range(1, n_harm + 1):
        feats.append(np.sin(2 * np.pi * k * x / p))
        feats.append(np.cos(2 * np.pi * k * x / p))
    return np.stack(feats, axis=1)


# ---------------------------------------------------------------------------
# one small neural net (hand-written, Adam) as an sklearn-compatible estimator
# ---------------------------------------------------------------------------
from sklearn.base import BaseEstimator, ClassifierMixin


class SmallNet(BaseEstimator, ClassifierMixin):
    """2-hidden-layer tanh MLP, full-batch Adam, logistic output. Deterministic."""

    def __init__(self, hidden=64, lr=1e-2, epochs=3000, wd=1e-4, seed=0):
        self.hidden, self.lr, self.epochs, self.wd, self.seed = hidden, lr, epochs, wd, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, float); y = np.asarray(y, float).reshape(-1, 1)
        d, h = X.shape[1], self.hidden
        W = [rng.normal(0, np.sqrt(1 / d), (d, h)), rng.normal(0, np.sqrt(1 / h), (h, h)),
             rng.normal(0, np.sqrt(1 / h), (h, 1))]
        b = [np.zeros((1, h)), np.zeros((1, h)), np.zeros((1, 1))]
        prm = W + b
        m = [np.zeros_like(q) for q in prm]; v = [np.zeros_like(q) for q in prm]
        b1, b2, eps = 0.9, 0.999, 1e-8
        n = X.shape[0]
        for t in range(1, self.epochs + 1):
            a1 = np.tanh(X @ W[0] + b[0])
            a2 = np.tanh(a1 @ W[1] + b[1])
            z = a2 @ W[2] + b[2]
            pr = 1 / (1 + np.exp(-z))
            dz = (pr - y) / n
            gW2 = a2.T @ dz + self.wd * W[2]; gb2 = dz.sum(0, keepdims=True)
            da2 = (dz @ W[2].T) * (1 - a2 ** 2)
            gW1 = a1.T @ da2 + self.wd * W[1]; gb1 = da2.sum(0, keepdims=True)
            da1 = (da2 @ W[1].T) * (1 - a1 ** 2)
            gW0 = X.T @ da1 + self.wd * W[0]; gb0 = da1.sum(0, keepdims=True)
            grads = [gW0, gW1, gW2, gb0, gb1, gb2]
            for i, gq in enumerate(grads):
                m[i] = b1 * m[i] + (1 - b1) * gq
                v[i] = b2 * v[i] + (1 - b2) * gq ** 2
                mh = m[i] / (1 - b1 ** t); vh = v[i] / (1 - b2 ** t)
                prm[i] -= self.lr * mh / (np.sqrt(vh) + eps)
        self.W_, self.b_, self.classes_ = W, b, np.array([0, 1])
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        a1 = np.tanh(X @ self.W_[0] + self.b_[0])
        a2 = np.tanh(a1 @ self.W_[1] + self.b_[1])
        return (a2 @ self.W_[2] + self.b_[2] > 0).astype(int).ravel()


# ---------------------------------------------------------------------------
# model bank + hyperparameter grids
# ---------------------------------------------------------------------------
def model_grids(seed):
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    return {
        "svm_rbf": (SVC(), {
            "C": [0.1, 1.0, 10.0, 100.0, 1000.0],
            "gamma": ["scale", 1e-3, 1e-2, 1e-1, 1.0]}),                    # 25
        "random_forest": (RandomForestClassifier(random_state=seed, n_jobs=1), {
            "n_estimators": [200, 400],
            "max_depth": [None, 5, 12],
            "max_features": ["sqrt", None]}),                               # 12
        "grad_boost": (GradientBoostingClassifier(random_state=seed), {
            "n_estimators": [100, 300],
            "max_depth": [2, 3],
            "learning_rate": [0.05, 0.2]}),                                 # 8
        "mlp": (MLPClassifier(random_state=seed, max_iter=800), {
            "hidden_layer_sizes": [(64,), (256,), (128, 128), (256, 256)],
            "alpha": [1e-4, 1e-2],
            "learning_rate_init": [1e-3, 1e-2]}),                           # 16
        "knn": (KNeighborsClassifier(), {
            "n_neighbors": [1, 3, 5, 7, 11, 15, 21],
            "weights": ["uniform", "distance"]}),                           # 14
        "logistic": (LogisticRegression(max_iter=5000), {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}),                   # 6
        # CAP: 4 configs / 1500 epochs. The hand-written net is by far the slowest
        # family (pure-numpy full-batch Adam): a timing probe at p=47 measured
        # 144.7 s for the 8-config/3000-epoch grid vs <15 s for every other family.
        # This cap is a COMPUTE decision, not an accuracy one -- the probe's
        # 3000-epoch best inner-CV score was 0.478 (i.e. already at chance).
        "small_net": (SmallNet(seed=seed), {
            "hidden": [64, 256],
            "lr": [1e-3, 1e-2],
            "epochs": [1500]}),                                             # 4
    }


def _tune(model, grid, X, y, seed, nested):
    """Grid-search a model. Returns (best_params, best_inner_cv, nested_cv|nan)."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    cv = StratifiedKFold(n_splits=CV_FOLDS)      # same splitter as dlp_real._cv_accuracy
    pipe = Pipeline([("sc", StandardScaler()), ("m", model)])
    pgrid = {f"m__{k}": v for k, v in grid.items()}
    gs = GridSearchCV(pipe, pgrid, cv=cv, scoring="accuracy", n_jobs=1, refit=True)
    gs.fit(X, y)
    best = {k.replace("m__", ""): (list(v) if isinstance(v, tuple) else v)
            for k, v in gs.best_params_.items()}
    nest = float("nan")
    if nested:
        gs2 = GridSearchCV(pipe, pgrid, cv=StratifiedKFold(n_splits=CV_FOLDS),
                           scoring="accuracy", n_jobs=1, refit=True)
        nest = float(np.mean(cross_val_score(gs2, X, y, cv=cv, n_jobs=1)))
    return best, float(gs.best_score_), nest


# ---------------------------------------------------------------------------
# one (prime, seed) unit
# ---------------------------------------------------------------------------
def run_unit(p, seed, feat_kinds, out_dir, nested):
    part = os.path.join(out_dir, "parts_tuned", f"p{p}_s{seed}.json")
    if os.path.exists(part):
        return part, False
    t0 = time.time()
    g = _primitive_root(p)
    dl = dlog_table(p, g)
    s = int(np.random.default_rng(seed).integers(0, p - 1))
    xs, y = concept(p, g, s, dl)
    rec = {"prime": p, "generator": g, "seed": seed, "hidden_shift_s": s,
           "n_examples": int(p - 1), "models": []}
    for kind in feat_kinds:
        X = features(xs, p, kind)
        for name, (model, grid) in model_grids(seed).items():
            best, inner, nest = _tune(model, grid, X, y, seed, nested)
            rec["models"].append({"model": name, "feature_set": kind,
                                  "n_features": int(X.shape[1]),
                                  "best_config": best, "cv_acc_tuned": inner,
                                  "nested_cv_acc": nest})
    # quantum reference: UNTUNED logistic on the dlog feature (as in dlp_real.py)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xq = quantum_feature(xs, dl, p)
    qacc = float(np.mean(cross_val_score(
        make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=2000)),
        Xq, y, cv=StratifiedKFold(n_splits=CV_FOLDS))))
    rec["quantum_dlog_acc"] = qacc
    rec["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(part), exist_ok=True)
    with open(part, "w") as f:
        json.dump(rec, f)
    print(f"[unit] p={p} seed={seed} q={qacc:.3f} ({rec['seconds']}s)", flush=True)
    return part, True


# ---------------------------------------------------------------------------
# merge -> CSV
# ---------------------------------------------------------------------------
def merge(out_dir):
    pdir = os.path.join(out_dir, "parts_tuned")
    recs = []
    for fn in sorted(os.listdir(pdir)):
        if fn.endswith(".json"):
            with open(os.path.join(pdir, fn)) as f:
                recs.append(json.load(f))
    if not recs:
        return None
    by_p = {}
    for r in recs:
        by_p.setdefault(r["prime"], []).append(r)

    rows = []
    for p, rs in sorted(by_p.items()):
        q = np.array([r["quantum_dlog_acc"] for r in rs])
        keys = {(m["model"], m["feature_set"]) for r in rs for m in r["models"]}
        for name, kind in sorted(keys):
            ms = [m for r in rs for m in r["models"]
                  if m["model"] == name and m["feature_set"] == kind]
            tuned = np.array([m["cv_acc_tuned"] for m in ms])
            nest = np.array([m["nested_cv_acc"] for m in ms])
            cfgs = [json.dumps(m["best_config"], sort_keys=True) for m in ms]
            modal = max(set(cfgs), key=cfgs.count)
            rows.append({
                "prime": p, "generator": rs[0]["generator"], "n_examples": rs[0]["n_examples"],
                "n_seeds": len(rs), "model": name, "feature_set": kind,
                "n_features": ms[0]["n_features"],
                "modal_best_config": modal,
                "n_seeds_with_modal_config": cfgs.count(modal),
                "cv_acc_tuned_mean": round(float(tuned.mean()), 4),
                "cv_acc_tuned_std": round(float(tuned.std()), 4),
                "nested_cv_acc_mean": round(float(np.nanmean(nest)), 4),
                "nested_cv_acc_std": round(float(np.nanstd(nest)), 4),
                "quantum_dlog_acc_mean": round(float(q.mean()), 4),
                "quantum_dlog_acc_std": round(float(q.std()), 4),
                "gap_quantum_minus_tuned": round(float(q.mean() - tuned.mean()), 4),
                "gap_quantum_minus_nested": round(float(q.mean() - np.nanmean(nest)), 4),
            })
        # per-prime summary row: best tuned classical over ALL models/feature sets
        allm = [m for r in rs for m in r["models"]]
        agg = {}
        for m in allm:
            agg.setdefault((m["model"], m["feature_set"]), []).append(m)
        bt = max(float(np.mean([m["cv_acc_tuned"] for m in v])) for v in agg.values())
        bn = max(float(np.nanmean([m["nested_cv_acc"] for m in v])) for v in agg.values())
        btk = max(agg, key=lambda k: float(np.mean([m["cv_acc_tuned"] for m in agg[k]])))
        bnk = max(agg, key=lambda k: float(np.nanmean([m["nested_cv_acc"] for m in agg[k]])))
        rows.append({
            "prime": p, "generator": rs[0]["generator"], "n_examples": rs[0]["n_examples"],
            "n_seeds": len(rs), "model": "BEST_TUNED_CLASSICAL", "feature_set": "any",
            "n_features": "", "modal_best_config": f"tuned:{btk[0]}/{btk[1]} nested:{bnk[0]}/{bnk[1]}",
            "n_seeds_with_modal_config": "",
            "cv_acc_tuned_mean": round(bt, 4), "cv_acc_tuned_std": "",
            "nested_cv_acc_mean": round(bn, 4), "nested_cv_acc_std": "",
            "quantum_dlog_acc_mean": round(float(q.mean()), 4),
            "quantum_dlog_acc_std": round(float(q.std()), 4),
            "gap_quantum_minus_tuned": round(float(q.mean() - bt), 4),
            "gap_quantum_minus_nested": round(float(q.mean() - bn), 4),
        })
    pth = os.path.join(out_dir, "dlp_baselines_tuned.csv")
    with open(pth, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return pth, len(rows)


def write_comparison(out_dir, wide_csv="results/discrete_log/dlp_wide/dlp_wide.csv"):
    """Head-to-head: UNTUNED best classical (from the existing dlp_wide per-seed CSV,
    restricted to the SAME primes and SAME seeds) vs the tuned best, vs quantum.
    This is the number that answers 'were your baselines undertuned?'."""
    tuned_csv = os.path.join(out_dir, "dlp_baselines_tuned.csv")
    if not (os.path.exists(tuned_csv) and os.path.exists(wide_csv)):
        return None
    with open(tuned_csv) as f:
        trows = list(csv.DictReader(f))
    best = {r["prime"]: r for r in trows if r["model"] == "BEST_TUNED_CLASSICAL"}
    primes = {int(p) for p in best}
    seeds = {int(r["n_seeds"]) for r in best.values()}
    nseed = max(seeds)
    with open(wide_csv) as f:
        wrows = [r for r in csv.DictReader(f)
                 if int(r["prime"]) in primes and int(r["seed"]) < nseed]
    cls_keys = [k for k in wrows[0] if k.startswith("cls_")] if wrows else []
    out = []
    for p in sorted(primes):
        rs = [r for r in wrows if int(r["prime"]) == p]
        if not rs:
            continue
        per = {k: float(np.mean([float(r[k]) for r in rs])) for k in cls_keys}
        ub = max(per.values()); ubk = max(per, key=per.get)
        t = best[str(p)]
        out.append({
            "prime": p, "n_seeds": len(rs),
            "untuned_best_classical": round(ub, 4),
            "untuned_best_model": ubk.replace("cls_", ""),
            "tuned_best_classical": float(t["cv_acc_tuned_mean"]),
            "tuned_best_which": t["modal_best_config"],
            "tuned_best_nested": float(t["nested_cv_acc_mean"]),
            "lift_from_tuning": round(float(t["cv_acc_tuned_mean"]) - ub, 4),
            "quantum_dlog_acc": float(t["quantum_dlog_acc_mean"]),
            "gap_quantum_minus_tuned_best": float(t["gap_quantum_minus_tuned"]),
            "gap_quantum_minus_nested_best": float(t["gap_quantum_minus_nested"]),
        })
    pth = os.path.join(out_dir, "dlp_baselines_tuned_vs_untuned.csv")
    with open(pth, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    return pth


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--primes", type=int, nargs="+", default=[23, 47, 59, 83, 101, 127])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--features", nargs="+", default=["rich16", "full_fourier"])
    ap.add_argument("--no-nested", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--out", default="results/discrete_log")
    a = ap.parse_args(argv)
    if not a.merge_only:
        units = [(p, s) for p in a.primes for s in a.seeds]
        os.makedirs(os.path.join(a.out, "parts_tuned"), exist_ok=True)
        done = 0
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            futs = [ex.submit(run_unit, p, s, a.features, a.out, not a.no_nested)
                    for p, s in units]
            for fut in as_completed(futs):
                fut.result(); done += 1
                print(f"[dlp_baselines_tuned] {done}/{len(units)} units", flush=True)
    out = merge(a.out)
    if out:
        print(f"wrote {out[0]} ({out[1]} rows)")
        cmp_path = write_comparison(a.out)
        if cmp_path:
            print(f"wrote {cmp_path}")


if __name__ == "__main__":
    main()
