
from __future__ import annotations

import csv
import os

import numpy as np

OUT_DIR = os.path.join("results", "multitask_benchmark")
BASIS = "heron_rz_sx_x_cz"
COUPLING = "all_to_all"


def read_csv(name):
    with open(os.path.join(OUT_DIR, name)) as fh:
        return list(csv.DictReader(fh))


def fnum(s):
    return float(s) if s not in ("", None) else None


def main():
    comps = read_csv("resource_components.csv")
    struct = {r["group"]: r for r in read_csv("group_structure.csv")}

    # index: (group, component) -> row, restricted to the primary basis
    idx = {}
    for r in comps:
        if r["basis"] == BASIS and r["coupling"] == COUPLING:
            idx[(r["group"], r["component"])] = r

    groups = [r["group"] for r in read_csv("group_structure.csv")]

    # ---- per-group derived table ----------------------------------------
    rows = []
    for g in groups:
        s = struct[g]
        order = int(s["order_G"])
        ab = int(s["is_abelian"])
        nd = int(s["n_data_qubits"])
        nf = int(s["n_frame_qubits"])

        def twoq(comp):
            r = idx.get((g, comp))
            return fnum(r["twoq_gates_mean"]) if r else None

        def dep(comp):
            r = idx.get((g, comp))
            return fnum(r["depth_mean"]) if r else None

        sel_lad = twoq("select_V_ladder")
        sel_nai = twoq("select_V_naive")
        gft_gen = twoq("group_fourier_generic")
        gft_str = twoq("group_fourier_structured_qft")
        prep = twoq("prep_G")

        rows.append(dict(
            group=g, order_G=order, is_abelian=ab,
            n_frame_qubits=nf, n_data_qubits=nd,
            n_ancilla_ladder=int(s["n_ancilla_ladder"]),
            irrep_dims=s["irrep_dims"], max_irrep_dim=int(s["max_irrep_dim"]),
            prepG_twoq=f"{prep:.1f}",
            selectV_ladder_twoq=f"{sel_lad:.1f}",
            selectV_naive_twoq=f"{sel_nai:.1f}",
            selectV_ladder_twoq_per_group_element=f"{sel_lad/order:.2f}",
            naive_over_ladder_ratio=f"{sel_nai/sel_lad:.2f}" if sel_lad else "",
            gft_generic_twoq=f"{gft_gen:.1f}",
            gft_structured_qft_twoq=(f"{gft_str:.1f}" if gft_str is not None else ""),
            gft_generic_over_structured=(f"{gft_gen/gft_str:.2f}"
                                         if gft_str not in (None, 0) else ""),
            structured_qft_available=int(gft_str is not None),
            selectV_ladder_depth=f"{dep('select_V_ladder'):.1f}",
            modeA_vacuous=int(s["modeA_vacuous"]),
            accept_prob_haar_rep_only=s["accept_prob_haar_modeA_rep_only"],
            notes=("structured_QFT_unavailable_nonabelian_or_|G|_not_power_of_2"
                   if gft_str is None else ""),
        ))

    # ---- Q1: matched-pair contrast --------------------------------------
    matched = []
    pair = ("C8d", "D4")   # |G|=8, n_data=2, dim H_+=1 for both
    if all(p in struct for p in pair):
        a, b = struct[pair[0]], struct[pair[1]]
        for comp in ("select_V_ladder", "select_V_naive",
                     "group_fourier_generic", "prep_G"):
            va = fnum(idx[(pair[0], comp)]["twoq_gates_mean"])
            vb = fnum(idx[(pair[1], comp)]["twoq_gates_mean"])
            matched.append(dict(
                comparison="matched_order8_ndata2_abelian_vs_nonabelian",
                component=comp,
                abelian_group=pair[0], nonabelian_group=pair[1],
                order_G=8, n_data_qubits=2,
                abelian_twoq=f"{va:.1f}", nonabelian_twoq=f"{vb:.1f}",
                ratio_nonabelian_over_abelian=(f"{vb/va:.3f}" if va else ""),
                confound_controlled="order_G,n_data_qubits,dim_Hplus",
                notes=(f"dimHplus_abelian={a['dim_Hplus_rep']} "
                       f"dimHplus_nonabelian={b['dim_Hplus_rep']}"),
            ))

    # ---- Q3: scaling of SELECT-V in |G| ---------------------------------
    # Fit log2(twoq) = alpha * log2(|G|) + c over groups sharing n_data = 2,
    # so the fit is not confounded by data-register width.
    scaling = []
    for nd_fix in (2,):
        sub = [r for r in rows if r["n_data_qubits"] == nd_fix]
        x = np.log2([r["order_G"] for r in sub])
        y = np.log2([float(r["selectV_ladder_twoq"]) for r in sub])
        if len(sub) >= 3:
            alpha, c = np.polyfit(x, y, 1)
            yhat = alpha * x + c
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            scaling.append(dict(
                quantity="selectV_ladder_twoq_vs_group_order",
                restricted_to=f"n_data_qubits={nd_fix}",
                n_groups=len(sub),
                groups="|".join(r["group"] for r in sub),
                fitted_exponent_alpha=f"{alpha:.3f}",
                r_squared=f"{r2:.4f}",
                interpretation=("cost ~ |G|^alpha; alpha near 1 means LINEAR in "
                                "|G| = EXPONENTIAL in log|G|, so 'for any finite "
                                "group' must be qualified"),
                poly_log_G=int(alpha < 0.25),
            ))

    # generic group Fourier transform vs 2**n_frame
    xf = np.log2([int(struct[r["group"]]["n_frame_qubits"]) and
                  2 ** int(struct[r["group"]]["n_frame_qubits"]) for r in rows])
    yf = np.log2([max(1.0, float(r["gft_generic_twoq"])) for r in rows])
    alpha_f, c_f = np.polyfit(xf, yf, 1)
    yhat = alpha_f * xf + c_f
    ss_res = float(np.sum((yf - yhat) ** 2))
    ss_tot = float(np.sum((yf - yf.mean()) ** 2))
    scaling.append(dict(
        quantity="group_fourier_generic_twoq_vs_2^n_frame",
        restricted_to="all_groups",
        n_groups=len(rows),
        groups="|".join(r["group"] for r in rows),
        fitted_exponent_alpha=f"{alpha_f:.3f}",
        r_squared=f"{1 - ss_res/ss_tot:.4f}" if ss_tot > 0 else "",
        interpretation=("generic (QSD) synthesis of the |G|x|G| Fourier "
                        "transform; alpha near 2 is the textbook 4^n_frame/4 "
                        "two-qubit scaling"),
        poly_log_G=0,
    ))

    def write(name, data):
        path = os.path.join(OUT_DIR, name)
        if os.path.exists(path):
            raise SystemExit(f"REFUSING TO OVERWRITE existing file: {path}")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
        print("wrote", path, f"({len(data)} rows)")

    write("resource_derived_per_group.csv", rows)
    write("abelian_vs_nonabelian.csv", matched)
    write("scaling_summary.csv", scaling)

    for r in rows:
        print(f"{r['group']:4s} |G|={r['order_G']:2d} ab={r['is_abelian']} "
              f"nd={r['n_data_qubits']} selectLadder2q={r['selectV_ladder_twoq']:>8s} "
              f"perElem={r['selectV_ladder_twoq_per_group_element']:>6s} "
              f"gftGen={r['gft_generic_twoq']:>6s} "
              f"gftQFT={r['gft_structured_qft_twoq']:>6s}")
    for m in matched:
        print(f"MATCHED {m['component']:22s} abelian={m['abelian_twoq']:>8s} "
              f"nonabelian={m['nonabelian_twoq']:>8s} "
              f"ratio={m['ratio_nonabelian_over_abelian']}")
    for s in scaling:
        print(f"SCALING {s['quantity']}: alpha={s['fitted_exponent_alpha']} "
              f"R2={s['r_squared']}")


if __name__ == "__main__":
    main()
