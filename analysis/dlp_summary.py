"""Discrete-log numbers quoted in the article, recomputed from the recorded files.

The gaps between the quantum feature and the classical models take the best
model in each (prime, seed) unit and average over seeds per prime, which is how
the article reports them. The BEST_TUNED_CLASSICAL rows of
results/discrete_log/dlp_baselines_tuned.csv take the best per-prime mean
instead and give larger gaps.

Run from the archive root::

    python -m analysis.dlp_summary
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict

import numpy as np

from cqfa.paths import RESULTS
from experiments.discrete_log.dlp_real import _primitive_root, concept, dlog_table

DLP = RESULTS / "discrete_log"
SEEDS = range(5)        # seeds of the recorded dlp_real run, the driver's default


def gaps():
    tuned, nested = defaultdict(list), defaultdict(list)
    for path in sorted((DLP / "parts_tuned").glob("p*_s*.json")):
        rec = json.loads(path.read_text())
        q, models = rec["quantum_dlog_acc"], rec["models"]
        tuned[rec["prime"]].append(q - max(m["cv_acc_tuned"] for m in models))
        nest = [m["nested_cv_acc"] for m in models
                if m["nested_cv_acc"] is not None and not math.isnan(m["nested_cv_acc"])]
        nested[rec["prime"]].append(q - max(nest))
    return tuned, nested


def adjacency():
    with open(DLP / "dlp_real.csv") as fh:
        primes = sorted({int(r["prime"]) for r in csv.DictReader(fh)})
    per_prime = []
    for p in primes:
        g = _primitive_root(p)
        dl = dlog_table(p, g)
        agree = []
        for seed in SEEDS:
            s = int(np.random.default_rng(seed).integers(0, p - 1))    # as in dlp_real.run
            _, y = concept(p, g, s, dl)
            agree.append(np.mean(y[:-1] == y[1:]))
        per_prime.append(np.mean(agree))
    return primes, float(np.mean(per_prime))


def main():
    tuned, nested = gaps()
    span = lambda xs: f"{min(xs):.3f} to {max(xs):.3f}"
    per_prime = lambda d: [float(np.mean(v)) for v in d.values()]
    print(f"primes {sorted(tuned)}, {sum(map(len, tuned.values()))} (prime, seed) units")
    print(f"quantum minus best tuned model, per-prime mean    {span(per_prime(tuned))}")
    print(f"quantum minus best tuned model, per seed          {span([g for v in tuned.values() for g in v])}")
    print(f"quantum minus best nested-CV model, per-prime     {span(per_prime(nested))}")
    with open(DLP / "dlp_baselines_tuned.csv") as fh:
        best23 = next(r for r in csv.DictReader(fh)
                      if r["model"] == "BEST_TUNED_CLASSICAL" and r["prime"] == "23")
    print(f"best tuned classical accuracy at p = 23           {float(best23['cv_acc_tuned_mean']):.3f}")
    primes, agree = adjacency()
    print(f"adjacent-x label agreement, p = {primes}, seeds 0-4   {agree:.2f}")


if __name__ == "__main__":
    main()
