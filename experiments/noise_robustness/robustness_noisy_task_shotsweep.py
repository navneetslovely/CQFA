"""Shot-budget sweep of the Mode-F noisy task.

At the 500-shot budget of robustness_noisy_task the Bayes ceiling of the Mode-F
task moves by only about 0.04 across channels, too little to test the
pre-registered criterion. This sweep repeats the Mode-F cells at shot budgets
5, 10, 25, 100 and 500 (five channels, six strengths, eight seeds) and applies
the same criterion, with a paired standard error for the excess shortfall.

Run from the archive root::

    python -m experiments.noise_robustness.robustness_noisy_task_shotsweep

Writes results/robustness/noisy_task_shotsweep_probe.csv and
results/robustness/noisy_task_shotsweep_verdict_probe.csv.
"""
import csv
import math
import numpy as np
from experiments.noise_robustness.robustness_noisy_task import run_modef_cell, _mse

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
CHANNELS = ["depolarizing", "dephasing", "amplitude_damping", "phase_damping", "coherent_z"]
STRENGTHS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
SHOTS = [5, 10, 25, 100, 500]
N, M, POOL = 8, 40, 400

rows = []
for shots in SHOTS:
    for c in CHANNELS:
        for s in STRENGTHS:
            per, info = run_modef_cell(c, s, "clean_state", N, M, shots, POOL, SEEDS)
            cq = [p[0] for p in per]; inf = [p[1] for p in per]; by = [p[2] for p in per]
            mc, sc = _mse(cq); mi, si = _mse(inf); mb, sb = _mse(by)
            # paired per-seed shortfalls (rule 5: difference first, then average)
            Mps = [b - a for b, a in zip(by, cq)]
            Ips = [b - a for b, a in zip(by, inf)]
            rows.append(dict(task="modef", channel=c, nominal_strength=s, shots=shots,
                             label_convention="clean_state", n_seeds=len(SEEDS),
                             label_signal_scale=info["label_signal_scale"],
                             majority_baseline=info["majority_baseline"],
                             bayes_acc=mb, bayes_se=sb, informed_acc=mi, informed_se=si,
                             cqfa_acc=mc, cqfa_se=sc,
                             M_perseed=";".join("%.6f" % x for x in Mps),
                             I_perseed=";".join("%.6f" % x for x in Ips)))
            print(f"shots={shots:<4} {c:<18} s={s:<5} bayes={mb:.4f} inf={mi:.4f} cqfa={mc:.4f}")

# apply the pre-registered criterion verbatim, with a PROPERLY PAIRED SE for dM
base = {(r["channel"], r["shots"]): r for r in rows if r["nominal_strength"] == 0.0}
vrows = []
for r in rows:
    b = base[(r["channel"], r["shots"])]
    Mp = [float(x) for x in r["M_perseed"].split(";")]
    Ip = [float(x) for x in r["I_perseed"].split(";")]
    M0 = [float(x) for x in b["M_perseed"].split(";")]
    I0 = [float(x) for x in b["I_perseed"].split(";")]
    dM_ps = [a - c for a, c in zip(Mp, M0)]     # paired per-seed dM
    dI_ps = [a - c for a, c in zip(Ip, I0)]
    dM = float(np.mean(dM_ps)); dI = float(np.mean(dI_ps))
    se_paired = float(np.std(dM_ps, ddof=1) / math.sqrt(len(dM_ps)))
    se_shipped = float(np.std(Mp, ddof=1) / math.sqrt(len(Mp)))   # what the module uses
    if r["nominal_strength"] == 0.0:
        v = "reference_strength0"
    elif r["label_signal_scale"] < 1e-10 or r["majority_baseline"] >= 1 - 1e-12:
        v = "vacuous_excluded"
    elif dM > 0.10 and dI <= 0.05:
        v = "KILL" if (se_paired > 0 and dM > 2 * se_paired) else "kill_inequality_unresolved"
    else:
        v = "SURVIVES"
    vrows.append(dict(shots=r["shots"], channel=r["channel"],
                      nominal_strength=r["nominal_strength"],
                      bayes_acc=r["bayes_acc"], cqfa_acc=r["cqfa_acc"],
                      informed_acc=r["informed_acc"],
                      excess_method_shortfall_dM=dM, excess_informed_shortfall_dI=dI,
                      dM_paired_se=se_paired, dM_se_as_shipped=se_shipped,
                      cell_verdict_paired_se=v))

for shots in SHOTS:
    sub = [r for r in vrows if r["shots"] == shots]
    ok = [r for r in sub if r["cell_verdict_paired_se"] not in
          ("reference_strength0", "vacuous_excluded")]
    b = [r["bayes_acc"] for r in sub]
    print(f"\nshots={shots}: bayes_span={max(b)-min(b):.4f} "
          f"max_dM={max([r['excess_method_shortfall_dM'] for r in ok] or [float('nan')]):.4f} "
          f"KILL={sum(1 for r in ok if r['cell_verdict_paired_se']=='KILL')} "
          f"unresolved={sum(1 for r in ok if r['cell_verdict_paired_se']=='kill_inequality_unresolved')}")

for p, rr in (("results/robustness/noisy_task_shotsweep_probe.csv", rows),
              ("results/robustness/noisy_task_shotsweep_verdict_probe.csv", vrows)):
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rr[0].keys())); w.writeheader(); w.writerows(rr)
    print("wrote", p)
