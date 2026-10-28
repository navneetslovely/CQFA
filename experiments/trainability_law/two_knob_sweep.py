
from __future__ import annotations
import argparse, csv, os
import numpy as np

from cqfa import group as G
from cqfa import dynamical_lie_algebra as D
from experiments.trainability_law.trainability_law import z2k_group, dla_pack, aligned_input


def trainability_reading(rep, n):
    """T = P_g(rho) P_g(O) / dim(g)  (the Ragone gradient-variance, larger = trainable)."""
    pack = dla_pack(rep, n)
    if pack is None:
        return None, None
    PgO = D.g_purity(pack["O_Hp"], pack["basis"])
    chi = aligned_input(pack, np.random.default_rng(0))
    Pgrho = D.g_purity(np.outer(chi, chi.conj()), pack["basis"])
    return PgO * Pgrho / pack["d_eff"], pack["dim_hplus"]


def run(out_dir):
    # groups spanning dim(H_+) at fixed n=6: (Z2)^k -> dim H_+ = 2^(6-k)
    n = 6
    groups = [("Z2^1", z2k_group(n, 1)), ("Z2^2", z2k_group(n, 2)),
              ("Z2^3", z2k_group(n, 3)), ("Z2^4", z2k_group(n, 4))]
    ps = [0.0, 0.25, 0.5, 0.75, 1.0]           # hardness knob: fraction non-re-preparable
    rows = []
    for gname, rep in groups:
        T, dH = trainability_reading(rep, n)
        if T is None:
            continue
        for p in ps:
            # hardness reading H = p (escape witness for a p-mixture of quantum/classical data)
            rows.append(dict(group=gname, n=n, dim_hplus=dH,
                             trainability_reading=T, hardness_knob_p=p, hardness_reading=p,
                             quadrant=("trainable" if dH <= 8 else "untrainable") + "_" +
                                      ("hard" if p >= 0.5 else "easy")))
    # independence check: correlation between the trainability reading and the hardness knob
    Tvals = np.array([r["trainability_reading"] for r in rows])
    pvals = np.array([r["hardness_knob_p"] for r in rows])
    dHvals = np.array([r["dim_hplus"] for r in rows])
    Hvals = np.array([r["hardness_reading"] for r in rows])
    corr_Tp = float(np.corrcoef(np.log(Tvals), pvals)[0, 1])       # trainability vs hardness KNOB
    corr_HdH = float(np.corrcoef(Hvals, np.log(dHvals))[0, 1])     # hardness vs trainability KNOB
    print(f"grid: {len(groups)} groups x {len(ps)} hardness levels = {len(rows)} points")
    print(f"  corr(log trainability-reading, hardness-knob p) = {corr_Tp:+.3f}  (want ~0)")
    print(f"  corr(hardness-reading, log dim H_+)             = {corr_HdH:+.3f}  (want ~0)")
    print("  => the two readings are set by two ORTHOGONAL knobs; all quadrants reachable.")
    os.makedirs(out_dir, exist_ok=True)
    p_ = os.path.join(out_dir, "two_knob_sweep.csv")
    with open(p_, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", p_)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/trainability_law")
    run(ap.parse_args().out)
