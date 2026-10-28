
import os
import numpy as np

from experiments.two_copy_separation.learning_separation_scaled import (
    make_dataset, ShadowAccumulator, _accuracy_at_threshold)

_ROOT = os.path.abspath(__file__)
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(
        os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)

TARGET, M, SEEDS = 0.9, 24, [0, 1, 2, 3, 4]


def acc_oracle(feats, labels, n):
    """Fixed threshold at the midpoint of the two TRUE purities. No fitting."""
    t = 0.5 * (1.0 + 2.0 ** -n)
    return float(np.mean((np.asarray(feats) > t) == np.asarray(labels)))


def acc_heldout(feats, labels, rng):
    """Fit best threshold on half, report accuracy on the other half."""
    f, y = np.asarray(feats, float), np.asarray(labels)
    idx = rng.permutation(len(f)); tr, te = idx[: len(f) // 2], idx[len(f) // 2:]
    fs = np.sort(f[tr]); best_t, best_a, best_p = None, -1, 1
    for t in (fs[:-1] + fs[1:]) / 2:
        for p in (1, -1):
            a = np.mean(((f[tr] > t) if p == 1 else (f[tr] <= t)) == y[tr])
            if a > best_a: best_a, best_t, best_p = a, t, p
    if best_t is None: return 0.5
    return float(np.mean(((f[te] > best_t) if best_p == 1 else (f[te] <= best_t)) == y[te]))


def copies_needed(n, sd, mode, max_c=1 << 14):
    rng = np.random.default_rng(sd)
    data = make_dataset(n, M, rng)
    labels = [y for _, y in data]
    accs = [ShadowAccumulator(st, rng) for st, _ in data]
    C = 4
    while C <= max_c:
        feats = [a.grow_to(C) for a in accs]
        if mode == "insample":
            a = _accuracy_at_threshold(feats, labels)
        elif mode == "oracle":
            a = acc_oracle(feats, labels, n)
        else:
            a = np.mean([acc_heldout(feats, labels, np.random.default_rng(9000 + k))
                         for k in range(20)])
        if a >= TARGET:
            return C
        C *= 2
    return np.nan


import csv, os
OUT = os.path.join(_ROOT, "results", "two_copy_separation")
rows, fits = [], []


if __name__ == "__main__":
    for mode in ("insample", "oracle", "heldout"):
        xs, ys = [], []
        for n in range(2, 8):
            v = [copies_needed(n, sd, mode) for sd in SEEDS]
            med = float(np.nanmedian(v))
            print(f"  {mode:9s} n={n}: per-seed={v}  median={med}", flush=True)
            rows.append(dict(scoring_rule=mode, n=n, seeds=len(SEEDS),
                             singlecopy_copies_median=med,
                             per_seed=";".join(str(int(z)) for z in v)))
            if not np.isnan(med): xs.append(n); ys.append(med)
        if len(xs) >= 3:
            s, b = np.polyfit(xs, np.log2(ys), 1)
            resid = np.log2(ys) - (s * np.array(xs) + b)
            ss = float(np.sum(resid ** 2)); st = float(np.sum((np.log2(ys) - np.mean(np.log2(ys))) ** 2))
            fits.append(dict(scoring_rule=mode, alpha_bits_per_qubit=round(float(s), 4),
                             base_per_qubit=round(float(2 ** s), 4),
                             intercept_log2=round(float(b), 4),
                             r2=round(1 - ss / st, 4),
                             max_abs_resid_bits=round(float(np.max(np.abs(resid))), 4),
                             n_min=min(xs), n_max=max(xs), n_points=len(xs)))
            print(f"  ==> {mode}: slope={s:.4f} bits/qubit, base={2**s:.3f}^n\n", flush=True)

    os.makedirs(OUT, exist_ok=True)
    for name, data in (("scoring_rule_robustness.csv", rows),
                       ("scoring_rule_robustness_fit.csv", fits)):
        p = os.path.join(OUT, name)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w.writeheader(); w.writerows(data)
        print("wrote", p)
