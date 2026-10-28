
from __future__ import annotations

import csv
import os
import sys

import numpy as np

_ROOT = os.path.abspath(__file__)
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(
        os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from experiments.readout_basis import u2_symmetry_discovery as u2  # noqa: E402

OUT = os.path.join(u2.ROOT, "results", "cqfa_pp")
SIGMA, N_TRAIN = u2.PRIMARY_CELL


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for true_name, H, _, _ in u2.SUBGROUP_PAIRS:
        Pi_true = u2.operator(u2.uniform_p_on(H))
        hidx = [u2.INDEX[h] for h in H]
        for seed in u2.SEEDS:
            rng = np.random.default_rng(1000 * seed + 7)
            T = rng.standard_normal((2, u2.ORDER)) @ Pi_true.T
            T /= np.linalg.norm(T, axis=1, keepdims=True)
            u2.TEMPLATES = T
            drng = np.random.default_rng(700000 + 1000 * seed + int(10 * SIGMA))
            Xtr, ytr, _ = u2.make_data(H, SIGMA, N_TRAIN, drng)

            frng = np.random.default_rng(99000 + seed)
            idx = frng.permutation(N_TRAIN)
            nval = max(4, int(u2.VAL_FRAC * N_TRAIN))
            vi, fi = idx[:nval], idx[nval:]
            Xf, yf, Xv, yv = Xtr[fi], ytr[fi], Xtr[vi], ytr[vi]

            for l2 in u2.L2_GRID:
                irng = np.random.default_rng(4242 + seed)
                inits = [np.ones(u2.ORDER)] + [
                    np.abs(irng.standard_normal(u2.ORDER)) + 1e-3 for _ in range(u2.N_RESTARTS - 1)
                ]
                for ri, t0 in enumerate(inits):
                    v0 = np.concatenate([t0, np.zeros(u2.ORDER + 1)])
                    p, w, b, floss = u2.fit_joint(Xf, yf, l2, v0)
                    val = (u2.predict(Xv @ u2.operator(p).T, w, b) == yv).mean()
                    rows.append(dict(
                        true_subgroup=true_name, true_order=len(H), seed=seed, l2=l2,
                        restart=ri, init="uniform" if ri == 0 else "random",
                        eff_support=float(1.0 / (p ** 2).sum()),
                        mass_H=float(p[hidx].sum()), p_max=float(p.max()),
                        train_loss=float(floss), val_acc=float(val),
                    ))
                # frozen oracle frame at the same L2, same split
                Zf = Xf @ Pi_true.T
                wo, bo = u2.fit_head(Zf, yf, l2)
                lo, _ = u2._loss_grad_head(wo, bo, Zf, yf, l2)
                rows.append(dict(
                    true_subgroup=true_name, true_order=len(H), seed=seed, l2=l2,
                    restart=-1, init="ORACLE_frozen",
                    eff_support=float(1.0 / (u2.uniform_p_on(H) ** 2).sum()),
                    mass_H=1.0, p_max=float(1.0 / len(H)),
                    train_loss=float(lo),
                    val_acc=float((u2.predict(Xv @ Pi_true.T, wo, bo) == yv).mean()),
                ))
        print(f"[probe] done {true_name}", flush=True)

    path = os.path.join(OUT, "u2_restart_collapse_probe.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[probe] wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
