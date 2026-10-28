
from __future__ import annotations

import csv
import math
import os
import statistics as st

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "results", "cqfa_pp", "t2_nonabelian_learned_readout.csv")
OUT = os.path.join(REPO, "results", "cqfa_pp", "t2_baseline_deltas.csv")

VACUOUS_TOL = 1e-10


def paired(sub, a, b):
    """Mean and standard error of the per-seed paired difference a - b."""
    d = [float(r[a]) - float(r[b]) for r in sub]
    m = st.mean(d)
    se = st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0.0
    return m, se


def main():
    rows = list(csv.DictReader(open(SRC)))
    keys = []
    for r in rows:
        k = (r["group"], r["task"])
        if k not in keys:
            keys.append(k)

    out = []
    for group, task in keys:
        sub = [r for r in rows if r["group"] == group and r["task"] == task]
        sig = st.mean([float(r["label_signal_scale"]) for r in sub])
        vacuous = sig < VACUOUS_TOL

        lm, lse = paired(sub, "acc_learned", "acc_modeF")
        bm, bse = paired(sub, "acc_modeF_bestbasis", "acc_modeF")
        xm, xse = paired(sub, "acc_learned", "acc_modeF_bestbasis")

        # fraction of the headline gap that a FIXED basis change already buys.
        # undefined when the headline gap is ~0 (the abelian controls).
        frac = bm / lm if abs(lm) > 1e-9 else float("nan")

        def verdict(m, se):
            if vacuous:
                return "vacuous_excluded"
            return "SURVIVES" if m > 2 * se else "negative"

        out.append(dict(
            group=group, task=task, abelian=sub[0]["abelian"], n_seeds=len(sub),
            mean_acc_modeF=st.mean([float(r["acc_modeF"]) for r in sub]),
            mean_acc_modeF_bestbasis=st.mean([float(r["acc_modeF_bestbasis"]) for r in sub]),
            mean_acc_learned=st.mean([float(r["acc_learned"]) for r in sub]),
            # the originally reported (weak-baseline) delta
            learned_minus_modeF=lm, learned_minus_modeF_se=lse,
            verdict_vs_modeF=verdict(lm, lse),
            # what a fixed basis change alone buys, with no learning at all
            bestbasis_minus_modeF=bm, bestbasis_minus_modeF_se=bse,
            verdict_bestbasis_vs_modeF=verdict(bm, bse),
            # the NUMBER: what training the readout adds over the best fixed basis
            learned_minus_bestbasis=xm, learned_minus_bestbasis_se=xse,
            verdict_vs_bestbasis=verdict(xm, xse),
            fixed_basis_fraction_of_gap=frac,
            label_signal_scale=sig, cell_is_vacuous=int(vacuous),
        ))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print(f"{'group/task':<16} {'learned-modeF':>17} {'bestbasis-modeF':>17} "
          f"{'learned-bestbasis':>19}  {'fixed%':>7}")
    for r in out:
        star = lambda v: "*" if v == "SURVIVES" else (" " if v == "negative" else "v")
        f = r["fixed_basis_fraction_of_gap"]
        fs = "  n/a " if math.isnan(f) or abs(r["learned_minus_modeF"]) < 1e-3 else f"{f*100:5.0f}%"
        print(f"{r['group']+'/'+r['task']:<16} "
              f"{r['learned_minus_modeF']:+8.4f}+-{r['learned_minus_modeF_se']:.4f}{star(r['verdict_vs_modeF'])} "
              f"{r['bestbasis_minus_modeF']:+8.4f}+-{r['bestbasis_minus_modeF_se']:.4f}{star(r['verdict_bestbasis_vs_modeF'])} "
              f"{r['learned_minus_bestbasis']:+9.4f}+-{r['learned_minus_bestbasis_se']:.4f}{star(r['verdict_vs_bestbasis'])} "
              f"{fs}")
    print("\n  * exceeds 2*se   v vacuous_excluded (no label signal)")
    print("  'fixed%' = share of the learned-vs-modeF gap already bought by a FIXED basis change.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
