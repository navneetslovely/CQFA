
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
    for true_name, H, wrong_name, Hw in u2.SUBGROUP_PAIRS:
        Pi_true = u2.operator(u2.uniform_p_on(H))
        hidx = [u2.INDEX[h] for h in H]
        for seed in u2.SEEDS:
            rng = np.random.default_rng(1000 * seed + 7)
            T = rng.standard_normal((2, u2.ORDER)) @ Pi_true.T
            T /= np.linalg.norm(T, axis=1, keepdims=True)
            u2.TEMPLATES = T

            trng = np.random.default_rng(500000 + 1000 * seed + int(10 * SIGMA))
            Xte, yte, _ = u2.make_data(H, SIGMA, u2.N_TEST, trng)
            drng = np.random.default_rng(700000 + 1000 * seed + int(10 * SIGMA))
            Xtr, ytr, _ = u2.make_data(H, SIGMA, N_TRAIN, drng)

            frng = np.random.default_rng(99000 + seed)
            acc_unaided, p_unaided, l2_h = u2.run_discovered_arm(Xtr, ytr, Xte, yte, frng)

            # same L2 as the search chose, so the two are comparable
            p_or = u2.uniform_p_on(H)
            theta_or = np.sqrt(p_or) + 1e-6
            v0 = np.concatenate([theta_or, np.zeros(u2.ORDER + 1)])
            p_fin, w_fin, b_fin, loss_fin = u2.fit_joint(Xtr, ytr, l2_h, v0)
            acc_oracle_init = (u2.predict(Xte @ u2.operator(p_fin).T, w_fin, b_fin) == yte).mean()

            # train loss of the frame at the same L2 (refit head, frame frozen)
            Ztr_h = Xtr @ u2.operator(p_unaided).T
            w_h, b_h = u2.fit_head(Ztr_h, ytr, l2_h)
            loss_unaided, _ = u2._loss_grad_head(w_h, b_h, Ztr_h, ytr, l2_h)
            # train loss at the FROZEN oracle frame
            Ztr_o = Xtr @ Pi_true.T
            w_o, b_o = u2.fit_head(Ztr_o, ytr, l2_h)
            loss_oracle_frozen, _ = u2._loss_grad_head(w_o, b_o, Ztr_o, ytr, l2_h)
            acc_oracle_frozen = (u2.predict(Xte @ Pi_true.T, w_o, b_o) == yte).mean()

            # where does the frame actually put its mass?
            part = float((p_unaided ** 2).sum())
            rows.append(dict(
                true_subgroup=true_name, seed=seed, sigma=SIGMA, n_train=N_TRAIN, l2=l2_h,
                p_identity_unaided=float(p_unaided[0]),
                p_max_unaided=float(p_unaided.max()),
                eff_support_unaided=float(1.0 / part),      # participation ratio, 1=collapsed, 24=uniform
                train_acc_unaided=float((u2.predict(Xtr @ u2.operator(p_unaided).T,
                                                   *u2.fit_head(Xtr @ u2.operator(p_unaided).T,
                                                                ytr, l2_h)) == ytr).mean()),
                train_acc_oracle_frozen=float((u2.predict(Xtr @ Pi_true.T,
                                                          *u2.fit_head(Xtr @ Pi_true.T,
                                                                       ytr, l2_h)) == ytr).mean()),
                mass_H_unaided=float(p_unaided[hidx].sum()),
                mass_H_after_oracle_init=float(p_fin[hidx].sum()),
                train_loss_unaided=float(loss_unaided),
                train_loss_oracle_frozen=float(loss_oracle_frozen),
                train_loss_after_oracle_init=float(loss_fin),
                acc_unaided=float(acc_unaided),
                acc_after_oracle_init=float(acc_oracle_init),
                acc_oracle_frozen=float(acc_oracle_frozen),
            ))
        print(f"[diag] done {true_name}", flush=True)

    path = os.path.join(OUT, "u2_oracle_init_diagnostic.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[diag] wrote {path} ({len(rows)} rows)")

    # readback summary
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[r["true_subgroup"]].append(r)
    print(f"\n{'H':>4}{'massH_hon':>11}{'massH_orIn':>12}{'p_ident':>9}{'effSupp':>9}"
          f"{'trL_hon':>10}{'trL_orFroz':>12}{'trAcc_hon':>11}{'trAcc_or':>10}"
          f"{'acc_hon':>9}{'acc_orIn':>10}{'acc_orFroz':>12}")
    for k, v in by.items():
        m = lambda key: float(np.mean([r[key] for r in v]))
        print(f"{k:>4}{m('mass_H_unaided'):>11.3f}{m('mass_H_after_oracle_init'):>12.3f}"
              f"{m('p_identity_unaided'):>9.3f}{m('eff_support_unaided'):>9.2f}"
              f"{m('train_loss_unaided'):>10.4f}{m('train_loss_oracle_frozen'):>12.4f}"
              f"{m('train_acc_unaided'):>11.3f}{m('train_acc_oracle_frozen'):>10.3f}"
              f"{m('acc_unaided'):>9.3f}{m('acc_after_oracle_init'):>10.3f}"
              f"{m('acc_oracle_frozen'):>12.3f}")


if __name__ == "__main__":
    main()
