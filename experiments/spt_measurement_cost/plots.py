
from __future__ import annotations

import csv
import os


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return float("nan")


def make_all(out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plots: matplotlib unavailable, skipping: {e}]")
        return

    cstar = _read_csv(os.path.join(out_dir, "cstar.csv"))
    ratio = _read_csv(os.path.join(out_dir, "ratio.csv"))
    strord = _read_csv(os.path.join(out_dir, "string_order.csv"))

    # -- headline plot ---------------------------------------------------------
    if cstar:
        arms = sorted(set(r["arm"] for r in cstar))
        plt.figure(figsize=(6, 4.2))
        for arm in arms:
            rows = sorted((r for r in cstar if r["arm"] == arm), key=lambda r: int(r["n"]))
            ns = [int(r["n"]) for r in rows]
            mean = [_f(r["cstar_mean"]) for r in rows]
            lo = [_f(r["cstar_ci_lo"]) for r in rows]
            hi = [_f(r["cstar_ci_hi"]) for r in rows]
            # skip inf points for the log plot
            xs, ys, los, his = [], [], [], []
            for n_, m_, l_, h_ in zip(ns, mean, lo, hi):
                if m_ == float("inf") or m_ != m_:
                    continue
                xs.append(n_); ys.append(m_); los.append(l_); his.append(h_)
            if xs:
                plt.plot(xs, ys, "o-", label=arm)
                plt.fill_between(xs, los, his, alpha=0.2)
        plt.yscale("log")
        plt.xlabel("system size n")
        plt.ylabel("C*  (copies to error <= epsilon)")
        plt.title("copies-to-target vs n (measurement-complexity separation)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "cstar_vs_n.png"), dpi=140)
        plt.close()

    # -- ratio plot ---------------------------------------------------------
    if ratio:
        rows = sorted(ratio, key=lambda r: int(r["n"]))
        ns = [int(r["n"]) for r in rows]
        rm = [_f(r["ratio_mean"]) for r in rows]
        rl = [_f(r["ratio_ci_lo"]) for r in rows]
        rh = [_f(r["ratio_ci_hi"]) for r in rows]
        xs, ys, ls, hs = [], [], [], []
        for n_, m_, l_, h_ in zip(ns, rm, rl, rh):
            if m_ != m_:
                continue
            xs.append(n_); ys.append(m_); ls.append(l_); hs.append(h_)
        if xs:
            plt.figure(figsize=(6, 4.2))
            plt.plot(xs, ys, "s-", color="C3")
            plt.fill_between(xs, ls, hs, alpha=0.2, color="C3")
            plt.axhline(1.0, color="k", lw=0.8, ls="--")
            plt.xlabel("system size n")
            plt.ylabel("C*_shadow_nonlinear / C*_coherent")
            plt.title("single-copy / 2-copy copy-budget ratio (grows with n)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "ratio_vs_n.png"), dpi=140)
            plt.close()

    # -- physics sanity -----------------------------------------------------
    if strord:
        ns = sorted(set(int(r["n"]) for r in strord))
        plt.figure(figsize=(6, 4.2))
        for n in ns:
            rows = sorted((r for r in strord if int(r["n"]) == n), key=lambda r: _f(r["lambda"]))
            lams = [_f(r["lambda"]) for r in rows]
            so = [_f(r["string_order"]) for r in rows]
            plt.plot(lams, so, "o-", label=f"n={n}")
        plt.axvline(1.0, color="k", lw=0.8, ls="--", label="transition")
        plt.xlabel("lambda (XX competitor strength)")
        plt.ylabel("string order  S = <prod K_i>")
        plt.title("SPT string order collapses across lambda ~ 1")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "string_order.png"), dpi=140)
        plt.close()

    print(f"[plots written to {out_dir}]")


if __name__ == "__main__":
    import sys
    make_all(sys.argv[1] if len(sys.argv) > 1 else "results/spt_measurement_cost/")
