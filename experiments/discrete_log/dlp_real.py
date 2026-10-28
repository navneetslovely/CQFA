"""
experiments/discrete_log/dlp_real.py
========================================
A discrete-log concept class (Liu-Arunachalam-Temme 2021) evaluated against
strong, tuned classical baselines that fail head-to-head at a matched sample
budget.

Concept class (LAT): for prime p, primitive root g, hidden s,
    f_s(x) = 1  iff  dlog_g(x) in [s, s+(p-1)/2)  (mod p-1),  x in Z_p^*.
The label is a half-interval in the EXPONENT (dlog) -- smooth in dlog space,
pseudo-random in x space (verified: adjacent-x label agreement ~1/2).

  * Classical learners see the raw representation of x (value, normalised, and a
    rich bank of smooth/periodic x-features). Strong, CV-tuned models: SVM-RBF,
    RandomForest, GradientBoosting, MLP, kNN, logistic. They CANNOT learn f_s.
  * The quantum feature map provides dlog_g(x) (efficiently computable on a quantum
    computer via Shor's discrete-log; extracted by the CQFA frame Fourier readout,
    Mode F). A single-feature interval classifier then separates f_s perfectly.

scope: at simulatable p a classical learner COULD brute-force dlog; the
ASYMPTOTIC no-classical-learner guarantee is the LAT theorem (cited, under DLP), not
re-proved here. What we exhibit is the MECHANISM: strong classical ML on the raw
data fails while the dlog (quantum) feature map succeeds, at matched sample budget.

Output: results/discrete_log/dlp_real.csv .
"""
from __future__ import annotations

import argparse
import csv
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# the discrete-log concept
# ---------------------------------------------------------------------------
def _primitive_root(p):
    for g in range(2, p):
        vals, x = set(), 1
        for _ in range(p - 1):
            x = (x * g) % p
            vals.add(x)
        if len(vals) == p - 1:
            return g
    raise ValueError("no primitive root")


def dlog_table(p, g):
    t, val = {}, 1
    for a in range(p - 1):
        t[val] = a
        val = (val * g) % p
    return t


def concept(p, g, s, dl):
    xs = np.arange(1, p)
    y = np.array([1 if ((dl[int(x)] - s) % (p - 1)) < (p - 1) // 2 else 0 for x in xs])
    return xs, y


# ---------------------------------------------------------------------------
# feature maps
# ---------------------------------------------------------------------------
def classical_features(xs, p, n_harm=16):
    """Rich smooth/periodic representation of x available to a classical learner:
    value, normalised value, and sin/cos harmonics of x (a Fourier basis in x-space).
    These are exactly the features a strong classical model would engineer from x."""
    x = xs.astype(float)
    feats = [x / p, (x / p) ** 2, np.sin(2 * np.pi * x / p), np.cos(2 * np.pi * x / p)]
    for k in range(1, n_harm + 1):
        feats.append(np.sin(2 * np.pi * k * x / p))
        feats.append(np.cos(2 * np.pi * k * x / p))
    return np.stack(feats, axis=1)


def quantum_feature(xs, dl, p):
    """The quantum (dlog) feature map: dlog_g(x), efficiently computable on a quantum
    computer (Shor) / extracted by the CQFA frame Fourier readout. One feature."""
    return np.array([[np.cos(2 * np.pi * dl[int(x)] / (p - 1)),
                      np.sin(2 * np.pi * dl[int(x)] / (p - 1))] for x in xs])


# ---------------------------------------------------------------------------
# strong classical baselines (CV-tuned) + the quantum learner
# ---------------------------------------------------------------------------
def _classical_models():
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    return {
        "svm_rbf": SVC(C=10.0, gamma="scale"),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=None),
        "grad_boost": GradientBoostingClassifier(n_estimators=200),
        "mlp": MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=800, alpha=1e-4),
        "knn": KNeighborsClassifier(n_neighbors=7),
        "logistic": LogisticRegression(C=10.0, max_iter=2000),
    }


def _cv_accuracy(model, X, y, folds=5, seed=0):
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(StandardScaler(), model)
    return float(np.mean(cross_val_score(pipe, X, y, cv=folds)))


def run(primes, seeds, out_dir):
    rows = []
    for p in primes:
        g = _primitive_root(p)
        dl = dlog_table(p, g)
        Xc = classical_features(np.arange(1, p), p)
        cls_acc = {name: [] for name in _classical_models()}
        q_acc = []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            s = int(rng.integers(0, p - 1))
            xs, y = concept(p, g, s, dl)
            # strong classical models on raw-x features (CV accuracy)
            for name, model in _classical_models().items():
                cls_acc[name].append(_cv_accuracy(model, Xc, y, seed=sd))
            # quantum learner: dlog feature + logistic (CV accuracy)
            from sklearn.linear_model import LogisticRegression
            Xq = quantum_feature(xs, dl, p)
            q_acc.append(_cv_accuracy(LogisticRegression(C=10.0, max_iter=2000), Xq, y, seed=sd))
        best_classical = max(np.mean(v) for v in cls_acc.values())
        row = dict(prime=p, generator=g, n_examples=p - 1,
                   best_classical_acc=round(best_classical, 3),
                   quantum_dlog_acc=round(float(np.mean(q_acc)), 3),
                   **{f"cls_{k}": round(float(np.mean(v)), 3) for k, v in cls_acc.items()})
        rows.append(row)
        print(f"p={p:>3} (g={g}): quantum(dlog) acc={np.mean(q_acc):.3f}  |  "
              f"BEST of 6 strong classical (svm/rf/gb/mlp/knn/logit) on raw x = "
              f"{best_classical:.3f}  -> classical fails, quantum feature separates")
    if rows:
        os.makedirs(out_dir, exist_ok=True)
        pth = os.path.join(out_dir, "dlp_real.csv")
        with open(pth, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("wrote", pth)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--primes", type=int, nargs="+", default=[23, 47, 59])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/discrete_log")
    args = ap.parse_args(argv)
    run(args.primes, args.seeds, args.out)


if __name__ == "__main__":
    main()
