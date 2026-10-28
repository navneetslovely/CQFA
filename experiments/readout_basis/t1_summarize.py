
from __future__ import annotations
import csv, os, sys
import numpy as np

_ROOT = os.path.abspath(__file__)
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(
        os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)
SRC = os.path.join(_ROOT, "results/readout_basis/t1_abelian_convergence.csv")
OUT = os.path.join(_ROOT, "results/readout_basis/t1_abelian_summary.csv")


def main():
    rows = list(csv.DictReader(open(SRC)))
    keys = sorted({(int(r["N"]), r["shots"]) for r in rows},
                  key=lambda t: (t[0], 10 ** 9 if t[1] == "exact" else int(t[1])))
    out = []
    for N, shots in keys:
        g = [r for r in rows if int(r["N"]) == N and r["shots"] == shots]
        aF = np.array([float(r["acc_modeF"]) for r in g])
        aL = np.array([float(r["acc_learned"]) for r in g])
        dd = np.array([float(r["d_to_DFT"]) for r in g])
        pF = np.array([float(r["acc_modeF_probs_only"]) for r in g])
        pL = np.array([float(r["acc_learned_probs_only"]) for r in g])
        vF = np.array([float(r["pop_shift_var_modeF"]) for r in g])
        vL = np.array([float(r["pop_shift_var_learned"]) for r in g])
        d = aL - aF
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        alarm = bool(d.mean() > 0 and d.mean() > 2 * se and se > 0) or bool(
            d.mean() > 0 and se == 0 and d.mean() > 1e-9)
        out.append(dict(
            N=N, shots=shots, n_seeds=len(g),
            acc_modeF_mean=round(float(aF.mean()), 4), acc_modeF_std=round(float(aF.std(ddof=1)), 4),
            acc_learned_mean=round(float(aL.mean()), 4), acc_learned_std=round(float(aL.std(ddof=1)), 4),
            delta_mean=round(float(d.mean()), 4), delta_paired_se=round(se, 4),
            ALARM_learned_beats_modeF=int(alarm),
            d_to_DFT_mean=round(float(dd.mean()), 4), d_to_DFT_std=round(float(dd.std(ddof=1)), 4),
            probs_only_acc_modeF=round(float(pF.mean()), 4),
            probs_only_acc_learned=round(float(pL.mean()), 4),
            pop_shift_var_modeF=float(f"{vF.mean():.3e}"),
            pop_shift_var_learned=float(f"{vL.mean():.3e}")))
        print(f"N={N:>3} shots={shots:>5} n={len(g)} | modeF={aF.mean():.3f}+-{aF.std(ddof=1):.3f} "
              f"learned={aL.mean():.3f}+-{aL.std(ddof=1):.3f} delta={d.mean():+.3f} (SE {se:.3f}) "
              f"ALARM={int(alarm)} | d_to_DFT={dd.mean():.3f}+-{dd.std(ddof=1):.3f} | "
              f"probs-only F={pF.mean():.3f} L={pL.mean():.3f} | popvar F={vF.mean():.1e} L={vL.mean():.1e}")
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print("wrote", OUT)
    nal = sum(r["ALARM_learned_beats_modeF"] for r in out)
    print(f"\nALARM fired in {nal}/{len(out)} (N, shots) cells.")
    ex = [r for r in out if r["shots"] == "exact"]
    print("EXACT-budget cells (the regime the gate theorem covers): "
          + "; ".join(f"N={r['N']} delta={r['delta_mean']:+.3f} ALARM={r['ALARM_learned_beats_modeF']}"
                      for r in ex))


if __name__ == "__main__":
    main()
