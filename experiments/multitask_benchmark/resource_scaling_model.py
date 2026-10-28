
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


def main():
    struct = {r["group"]: r for r in read_csv("group_structure.csv")}
    comps = {}
    for r in read_csv("resource_components.csv"):
        if r["basis"] == BASIS and r["coupling"] == COUPLING:
            comps[(r["group"], r["component"])] = r

    # ---------------- (1) mechanistic scaling model ----------------------
    rows = []
    for g, s in struct.items():
        order = int(s["order_G"])
        nf = int(s["n_frame_qubits"])
        twoq = float(comps[(g, "select_V_ladder")]["twoq_gates_mean"])
        # predicted structure of the ancilla-ladder SELECT
        n_terms = order - 1                       # identity term is skipped
        ladder_units = 2 * nf - 1                 # 2(nf-1) Toffolis + 1 cU_g
        denom = n_terms * ladder_units
        rows.append(dict(
            group=g, order_G=order, n_frame_qubits=nf,
            is_abelian=int(s["is_abelian"]),
            n_data_qubits=int(s["n_data_qubits"]),
            selectV_ladder_twoq=f"{twoq:.1f}",
            n_nonidentity_terms=n_terms,
            ladder_units_per_term=ladder_units,
            predicted_units=denom,
            twoq_per_unit=f"{twoq/denom:.3f}" if denom else "",
            included_in_cv=int(nf >= 2),
        ))

    vals = [float(r["twoq_per_unit"]) for r in rows if r["included_in_cv"]]
    cv = float(np.std(vals, ddof=1) / np.mean(vals))
    verdict = "CONFIRMS_|G|log|G|_model" if cv < 0.25 else "KILLS_|G|log|G|_model"
    rows.append(dict(
        group="ALL(n_frame>=2)", order_G="", n_frame_qubits="",
        is_abelian="", n_data_qubits="", selectV_ladder_twoq="",
        n_nonidentity_terms="", ladder_units_per_term="", predicted_units="",
        twoq_per_unit=f"mean={np.mean(vals):.3f} sd={np.std(vals, ddof=1):.3f} "
                      f"CV={cv:.3f} n={len(vals)} -> {verdict}",
        included_in_cv="",
    ))

    # ---------------- (2) confound check ---------------------------------
    conf = []
    for comp in ("single_U_g_worstcase", "select_V_naive", "select_V_ladder"):
        a = float(comps[("C8d", comp)]["twoq_gates_mean"])
        b = float(comps[("D4", comp)]["twoq_gates_mean"])
        conf.append(dict(
            component=comp,
            abelian_group="C8d", abelian_rep_structure="diagonal",
            nonabelian_group="D4", nonabelian_rep_structure="permutation",
            order_G=8, n_data_qubits=2,
            abelian_twoq=f"{a:.1f}", nonabelian_twoq=f"{b:.1f}",
            ratio_nonab_over_ab=(f"{b/a:.3f}" if a else "inf(abelian=0)"),
            dominated_by=("single U_g synthesis" if comp != "select_V_ladder"
                          else "frame address decoding (Toffoli ladder)"),
        ))
    lad = [c for c in conf if c["component"] == "select_V_ladder"][0]
    lad_ratio = float(lad["ratio_nonab_over_ab"])
    conf.append(dict(
        component="VERDICT", abelian_group="", abelian_rep_structure="",
        nonabelian_group="", nonabelian_rep_structure="",
        order_G="", n_data_qubits="", abelian_twoq="", nonabelian_twoq="",
        ratio_nonab_over_ab=f"{lad_ratio:.3f}",
        dominated_by=("KILLS_nonabelian_oracle_penalty: ladder ratio < 2 at "
                      "matched |G| and n_data; the large naive-SELECT ratio is "
                      "an artifact of diagonal-vs-permutation U_g structure"
                      if lad_ratio < 2 else
                      "SURVIVES_nonabelian_oracle_penalty"),
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

    write("scaling_model_fit.csv", rows)
    write("oracle_confound_check.csv", conf)

    for r in rows[:-1]:
        print(f"{r['group']:4s} |G|={r['order_G']:2d} nf={r['n_frame_qubits']} "
              f"2q={r['selectV_ladder_twoq']:>8s} units={r['predicted_units']:>4d} "
              f"per_unit={r['twoq_per_unit']:>7s} incl={r['included_in_cv']}")
    print(rows[-1]["twoq_per_unit"])
    for c in conf:
        print(f"{c['component']:22s} ab={c['abelian_twoq']:>8s} "
              f"nonab={c['nonabelian_twoq']:>8s} ratio={c['ratio_nonab_over_ab']}")


if __name__ == "__main__":
    main()
