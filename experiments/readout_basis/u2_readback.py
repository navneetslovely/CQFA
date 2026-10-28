
import csv
import os
import sys

ROOT = os.path.abspath(__file__)
while ROOT != os.path.dirname(ROOT) and not os.path.exists(
        os.path.join(ROOT, "pyproject.toml")):
    ROOT = os.path.dirname(ROOT)
P = os.path.join(ROOT, "results", "cqfa_pp", "u2_symmetry_discovery_summary.csv")


if __name__ == "__main__":
    rows = list(csv.DictReader(open(P)))
    f = lambda r, k: float(r[k])

    print("=== PRIMARY CELL (pre-registered sigma=2.0, n_train=80) ===")
    hdr = ("H", "massH", "massRnd", "gap", "2se_gap", "disc", "orac", "wrong", "triv", "unif",
           "d(or-di)", "d(di-wr)", "2se", "d(di-tr)", "2se", "verdict")
    print(("{:>4}" + "{:>9}" * 4 + "{:>7}" * 5 + "{:>9}" * 5 + "  {}").format(*hdr))
    for r in rows:
        if r["is_primary_cell"] != "1":
            continue
        print(("{:>4}" + "{:>9.3f}" * 4 + "{:>7.3f}" * 5 + "{:>9.3f}" * 5 + "  {}").format(
            r["true_subgroup"], f(r, "mean_mass_H_learned"), f(r, "mean_mass_H_random"),
            f(r, "mean_recovery_gap"), 2 * f(r, "se_recovery_gap"),
            f(r, "mean_acc_discovered"), f(r, "mean_acc_oracle"), f(r, "mean_acc_wrong"),
            f(r, "mean_acc_trivial"), f(r, "mean_acc_uniform_ambient"),
            f(r, "mean_oracle_minus_discovered"),
            f(r, "mean_discovered_minus_wrong"), 2 * f(r, "se_discovered_minus_wrong"),
            f(r, "mean_discovered_minus_trivial"), 2 * f(r, "se_discovered_minus_trivial"),
            r["verdict"]))

    print("\n=== FULL SWEEP verdict tally ===")
    tally = {}
    for r in rows:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    for k, v in sorted(tally.items()):
        print(f"  {k:30s} {v:3d} / {len(rows)}")

    print("\n=== ALL CELLS ===")
    print("{:>4}{:>7}{:>7}{:>9}{:>9}{:>8}{:>8}{:>8}{:>8}{:>10}  {}".format(
        "H", "sigma", "ntr", "massH", "gap", "disc", "orac", "wrong", "triv", "or-triv", "verdict"))
    for r in sorted(rows, key=lambda r: (r["true_subgroup"], float(r["sigma"]), int(r["n_train"]))):
        print("{:>4}{:>7}{:>7}{:>9.3f}{:>9.3f}{:>8.3f}{:>8.3f}{:>8.3f}{:>8.3f}{:>10.3f}  {}".format(
            r["true_subgroup"], r["sigma"], r["n_train"],
            f(r, "mean_mass_H_learned"), f(r, "mean_recovery_gap"),
            f(r, "mean_acc_discovered"), f(r, "mean_acc_oracle"), f(r, "mean_acc_wrong"),
            f(r, "mean_acc_trivial"), f(r, "mean_oracle_minus_trivial"), r["verdict"]))

    print("\n=== label_signal_scale range (vacuity guard, floor 1e-10) ===")
    lss = [f(r, "min_label_signal_scale") for r in rows]
    print(f"  min={min(lss):.6g}  max={max(lss):.6g}")
    print(f"  n_seeds per cell: {sorted({r['n_seeds'] for r in rows})}")
