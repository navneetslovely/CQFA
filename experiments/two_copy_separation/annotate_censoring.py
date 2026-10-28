
from __future__ import annotations

import csv
import os
import re

D = "results/two_copy_separation"
CSV = os.path.join(D, "learning_separation_scaled.csv")
LOG = os.path.join(D, "learning_separation_scaled_run.log")


def max_budget_per_n(log_path):
    """Largest C printed under each n block of the sweep stdout."""
    out, cur = {}, None
    for line in open(log_path):
        m = re.search(r"n=(\d+) seed=", line)
        if m:
            cur = int(m.group(1)); continue
        m = re.search(r"C=\s*(\d+)\s+acc=", line)
        if m and cur is not None:
            out[cur] = max(out.get(cur, 0), int(m.group(1)))
    return out


def main():
    mb = max_budget_per_n(LOG)
    rows = list(csv.DictReader(open(CSV)))
    for r in rows:
        n = int(r["n"])
        capped = int(r["singlecopy_capped_runs"])
        r["singlecopy_max_budget_tested"] = mb.get(n, "")
        r["singlecopy_is_lower_bound"] = 1 if capped > 0 else 0
    fields = list(rows[0].keys())
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    for r in rows:
        flag = " (LOWER BOUND)" if int(r["singlecopy_is_lower_bound"]) else ""
        print(f"n={r['n']:>2}  coherent={r['coherent_copies']:>4}  "
              f"single-copy={r['singlecopy_copies']:>9}{flag}  "
              f"ratio={r['ratio']:>9}  censored_seeds={r['singlecopy_capped_runs']}/{r['seeds']}"
              f"  max_C_tested={r['singlecopy_max_budget_tested']}")
    print("wrote", CSV)


if __name__ == "__main__":
    main()
