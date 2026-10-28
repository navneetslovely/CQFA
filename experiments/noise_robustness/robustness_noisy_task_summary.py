"""
experiments/robustness_noisy_task_summary.py
============================================
Aggregates results/robustness/noisy_task_redesign_v2.csv into the per-task summary
that the paper/rebuttal cites, so that EVERY reported number exists in a CSV cell
(integrity rule 1). Reads only; writes NEW files.

It also computes the POWER DIAGNOSTIC that decides whether the pre-registered
criterion actually had teeth on a given task:

    bayes_span = max(bayes_acc) - min(bayes_acc)  over non-vacuous cells

The criterion ("CQFA tracks the noisy-task Bayes accuracy") can only be TESTED
where the Bayes ceiling actually moves. If bayes_span is small the criterion is
passed vacuously and MUST be reported as underpowered, not as confirmation.
Threshold fixed here at 0.10 accuracy: below that the task never got materially
harder anywhere in the grid, so "the method degrades because the task degrades"
was never put at risk.

Outputs:
  results/robustness/noisy_task_summary_v2.csv
  results/robustness/noisy_task_channel_spread_v2.csv
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

POWER_MIN_BAYES_SPAN = 0.10


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/robustness/noisy_task_redesign_v2.csv")
    ap.add_argument("--out", default="results/robustness")
    ap.add_argument("--tag", default="_v2")
    args = ap.parse_args(argv)
    R = list(csv.DictReader(open(args.src)))
    f = lambda r, k: float(r[k])

    rows = []
    for task in ("modef", "twocopy"):
        for conv in ("clean_state", "noisy_state"):
            sel = [r for r in R if r["task"] == task and r["label_convention"] == conv]
            ok = [r for r in sel if r["vacuity_flag"] == "ok"]
            noisy = [r for r in ok if f(r, "nominal_strength") > 0]
            b = [f(r, "bayes_acc") for r in ok]
            q = [f(r, "cqfa_acc") for r in ok]
            i = [f(r, "informed_acc") for r in ok]
            span = (max(b) - min(b)) if b else float("nan")
            dM = [f(r, "excess_method_shortfall_dM") for r in noisy
                  if r["excess_method_shortfall_dM"] != ""]
            dI = [f(r, "excess_informed_shortfall_dI") for r in noisy
                  if r["excess_informed_shortfall_dI"] != ""]
            powered = span >= POWER_MIN_BAYES_SPAN
            rows.append(dict(
                task=task, label_convention=conv,
                n_cells=len(sel), n_nonvacuous=len(ok),
                n_single_class_degenerate=sum(
                    1 for r in sel if r["vacuity_flag"] == "single_class_degenerate"),
                n_near_degenerate_maj_ge_095=sum(
                    1 for r in ok if f(r, "majority_baseline") >= 0.95),
                bayes_min=min(b) if b else "", bayes_max=max(b) if b else "",
                bayes_span=span,
                cqfa_min=min(q) if q else "", cqfa_max=max(q) if q else "",
                informed_min=min(i) if i else "", informed_max=max(i) if i else "",
                corr_cqfa_bayes=(float(np.corrcoef(b, q)[0, 1]) if len(b) > 2 else ""),
                corr_informed_bayes=(float(np.corrcoef(b, i)[0, 1]) if len(b) > 2 else ""),
                max_excess_method_shortfall_dM=max(dM) if dM else "",
                max_excess_informed_shortfall_dI=max(dI) if dI else "",
                mean_label_flip_rate=float(np.mean(
                    [f(r, "label_flip_rate_vs_clean") for r in sel])),
                max_label_flip_rate=max(f(r, "label_flip_rate_vs_clean") for r in sel),
                criterion_powered=("yes" if powered else "no"),
                power_note=("bayes ceiling moves >= %.2f, criterion testable" % POWER_MIN_BAYES_SPAN
                            if powered else
                            "bayes ceiling span %.3f < %.2f: task never materially "
                            "degrades, criterion passed VACUOUSLY -- not confirmation"
                            % (span, POWER_MIN_BAYES_SPAN)),
            ))

    p = os.path.join(args.out, f"noisy_task_summary{args.tag}.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", p)
    for r in rows:
        print(f"{r['task']:<9}{r['label_convention']:<13} bayes_span={r['bayes_span']:.3f} "
              f"powered={r['criterion_powered']:<4} maxdM={r['max_excess_method_shortfall_dM']} "
              f"flip_max={r['max_label_flip_rate']:.3f} degen={r['n_single_class_degenerate']}")

    # physical spread at fixed nominal strength: equal s is NOT equal disturbance
    srows = []
    for s in sorted({f(r, "nominal_strength") for r in R}):
        sub = {}
        for r in R:
            if f(r, "nominal_strength") == s:
                sub[r["channel"]] = (f(r, "entanglement_fidelity"),
                                     f(r, "diamond_numeric_lower_bound"))
        if not sub:
            continue
        fe = [v[0] for v in sub.values()]; dm = [v[1] for v in sub.values()]
        srows.append(dict(
            nominal_strength=s, n_channels=len(sub),
            entanglement_fidelity_min=min(fe), entanglement_fidelity_max=max(fe),
            diamond_numeric_lower_min=min(dm), diamond_numeric_lower_max=max(dm),
            diamond_spread_ratio=(max(dm) / min(dm) if min(dm) > 0 else ""),
            argmin_diamond=min(sub, key=lambda k: sub[k][1]),
            argmax_diamond=max(sub, key=lambda k: sub[k][1]),
        ))
    p2 = os.path.join(args.out, f"noisy_task_channel_spread{args.tag}.csv")
    with open(p2, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0].keys())); w.writeheader(); w.writerows(srows)
    print("wrote", p2)
    for r in srows:
        print(f"s={r['nominal_strength']:<6} diamond_lower in "
              f"[{r['diamond_numeric_lower_min']:.4f} ({r['argmin_diamond']}), "
              f"{r['diamond_numeric_lower_max']:.4f} ({r['argmax_diamond']})] "
              f"ratio={r['diamond_spread_ratio']}")
    return rows


if __name__ == "__main__":
    main()
