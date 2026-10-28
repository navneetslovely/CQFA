
from __future__ import annotations
import os, glob, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figures.style import apply_style, new_fig, bold_ticks, thin_ticks, bar_labels, save, PALETTE, MARKERS  # noqa

from cqfa.paths import ROOT as _ARCHIVE_ROOT

ROOT = str(_ARCHIVE_ROOT)
R = lambda *p: os.path.join(ROOT, *p)


def _has(path):
    if not os.path.exists(path):
        print(f"  SKIP (missing {os.path.relpath(path, ROOT)})")
        return False
    return True


# ---------------------------------------------------------------- F2: measurement-complexity headline
def fig_measurement_complexity():
    print("F2 measurement-complexity separation")
    path = R("results/spt_measurement_cost/cstar.csv")
    if not _has(path):
        return
    df = pd.read_csv(path)
    ceiling = float(np.nanmax([b for b in df.get("epsilon", []) if False] + [16384]))
    apply_style()
    fig, ax = new_fig()
    arm_style = {  # order controls legend
        "shadow_nonlinear": ("single-copy (unknown)", PALETTE["single_copy"], "s", "-"),
        "coherent": ("coherent 2-copy", PALETTE["coherent"], "o", "-"),
        "shadow_known": ("single-copy (known)", PALETTE["neutral"], "^", "--"),
    }
    for arm, (lab, col, mk, ls) in arm_style.items():
        d = df[df["arm"] == arm].sort_values("n")
        if d.empty:
            continue
        y = d["cstar_mean"].to_numpy(dtype=float)
        lo = d.get("cstar_ci_lo", d["cstar_mean"]).to_numpy(dtype=float)
        hi = d.get("cstar_ci_hi", d["cstar_mean"]).to_numpy(dtype=float)
        n = d["n"].to_numpy(dtype=float)
        fin = np.isfinite(y)
        ax.plot(n[fin], y[fin], marker=mk, ls=ls, color=col, label=lab)
        if arm == "shadow_nonlinear":
            ax.fill_between(n[fin], np.clip(lo[fin], 1, None), np.clip(hi[fin], None, ceiling * 4),
                            color=col, alpha=0.15)
    # local observables: never reach target -> mark at the ceiling with an annotation
    dl = df[df["arm"] == "local"].sort_values("n")
    if not dl.empty:
        n = dl["n"].to_numpy(dtype=float)
        ax.plot(n, [ceiling] * len(n), marker="x", ls=":", color=PALETTE["augmented"],
                label="local obs. (fails)", markersize=10, markeredgewidth=3)
    ax.set_xlabel(r"system size $n$")
    ax.set_ylabel("copies to target")
    ax.set_yscale("log")
    ax.set_ylim(8, ceiling * 6)
    ax.legend(loc="center right", fontsize=10.5)
    bold_ticks(ax); thin_ticks(ax)
    save(fig, "f2_measurement_complexity")


# ---------------------------------------------------------------- F3: gradient variance
def fig_gradient_variance():
    print("F3 gradient-variance scaling")
    path = R("results/trainability_law/variance.csv")
    if not _has(path):
        return
    df = pd.read_csv(path)
    # pick the variance column and a single component if present
    varcol = next((c for c in ["var_mean", "grad_var_mean", "variance", "var"] if c in df.columns), None)
    if varcol is None:
        print("  SKIP (no variance column):", list(df.columns)); return
    if "component" in df.columns:
        comp = sorted(df["component"].unique())[0]
        df = df[df["component"] == comp]
    apply_style()
    fig, ax = new_fig()
    mstyle = {"cqfa": ("CQFA", PALETTE["cqfa"], "o"),
              "non_equivariant": ("generic ansatz", PALETTE["non_equivariant"], "s"),
              "classical_fa": ("classical FA", PALETTE["classical_fa"], "^")}
    for m, (lab, col, mk) in mstyle.items():
        d = df[df["model"] == m].sort_values("n")
        if d.empty:
            continue
        y = d[varcol].to_numpy(dtype=float)
        ax.plot(d["n"], y, marker=mk, color=col, label=lab)
        lo = next((c for c in ["ci_lo", "var_ci_lo"] if c in d.columns), None)
        hi = next((c for c in ["ci_hi", "var_ci_hi"] if c in d.columns), None)
        if lo and hi:
            ax.fill_between(d["n"], d[lo], d[hi], color=col, alpha=0.15)
    ax.set_xlabel(r"qubit count $n$")
    ax.set_ylabel(r"$\mathrm{Var}[\partial_\theta L]$")
    ax.set_yscale("log")
    ax.legend(loc="lower left")
    bold_ticks(ax); thin_ticks(ax)
    save(fig, "f3_gradient_variance")


# ---------------------------------------------------------------- F1: the decoupling dial
def fig_dial():
    print("F1 Representation Dial (decoupling: trainability vs hardness)")
    pq = R("results/trainability_law/certificate_points.csv")            # quantum (non-re-preparable)
    pc = R("results/trainability_law_classical/certificate_points.csv")  # classical (re-preparable) = trap
    if not _has(pq):
        return
    q = pd.read_csv(pq); q = q[q["n"] == q["n"].max()]
    apply_style()
    fig, ax = new_fig(w=4.0, h=3.1)
    # quadrant shading: x = DLA dim (small=trainable, left); y = hardness (escape, up=hard)
    ax.axhspan(0.5, 1.15, xmin=0, xmax=0.5, color="#0072B2", alpha=0.08)   # trainable & hard (goal)
    ax.axhspan(-0.15, 0.5, xmin=0, xmax=0.5, color="#D55E00", alpha=0.10)  # trainable & easy (trap)

    def pt(dframe, model, hardness, col, mk, note, dx, dy, ha="left"):
        g = dframe[dframe["model"] == model]
        if g.empty:
            return
        x = float(g["T_d_eff"].mean())
        ax.scatter([x], [hardness], s=260, color=col, marker=mk, zorder=5,
                   edgecolor="k", linewidth=1.3)
        ax.annotate(note, (x, hardness), xytext=(x * dx, hardness + dy), ha=ha,
                    fontsize=11, fontweight="bold", color=col, zorder=6)

    pt(q, "cqfa", 1.0, PALETTE["cqfa"], "*", "CQFA\n(quantum data)", 1.6, -0.06)
    pt(q, "non_equivariant", 1.0, PALETTE["non_equivariant"], "^", "generic\nansatz", 1.0, -0.30, ha="center")
    if os.path.exists(pc):
        c = pd.read_csv(pc); c = c[c["n"] == c["n"].max()]
        pt(c, "cqfa", 0.0, PALETTE["classical_fa"], "X", "CQFA\n(classical data)", 1.5, 0.05)
    ax.text(0.06, 0.93, "trainable & hard", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color="#0072B2")
    ax.text(0.06, 0.30, "the trap:\ntrainable & simulable", transform=ax.transAxes,
            fontsize=10, fontweight="bold", color="#B24000")
    ax.set_xlabel(r"$\dim\,\mathfrak{g}_G$ (trainability)")
    ax.set_ylabel("classically hard")
    ax.set_xscale("log")
    ax.set_xlim(3, 3000)
    ax.set_ylim(-0.2, 1.25)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["easy", "hard"])
    bold_ticks(ax)
    save(fig, "f1_dial")


# ---------------------------------------------------------------- F7: DLA dimension
def fig_dla():
    print("F7 DLA dimension scaling")
    path = R("results/trainability_law/dla.csv")
    if not _has(path):
        return
    df = pd.read_csv(path)
    gen = df[df["which"] == "generic_hea"].sort_values("n")
    cq = df[df["which"] == "cqfa_restricted"].sort_values("n")
    apply_style()
    fig, ax = new_fig()
    ax.plot(gen["n"], gen["dim_g_ambient"], marker="s", color=PALETTE["non_equivariant"], label="generic")
    ax.plot(cq["n"], cq["dim_g_restricted"], marker="o", color=PALETTE["cqfa"], label="CQFA")
    ax.set_xlabel(r"qubit count $n$")
    ax.set_ylabel(r"$\dim\,\mathfrak{g}$")
    ax.set_yscale("log")
    ax.legend(loc="center right")
    bold_ticks(ax); thin_ticks(ax)
    save(fig, "f7_dla_dimension")


# ---------------------------------------------------------------- F5: discrete-log
def fig_dlog():
    print("F5 discrete-log separation")
    cands = sorted(glob.glob(R("runs/run_dlog_p*_*/summary.csv")))
    if not cands:
        print("  SKIP (no discrete-log summary.csv)"); return
    df = pd.read_csv(cands[-1])
    r = df.iloc[0]
    apply_style()
    fig, ax = new_fig()
    labels = ["structured", "classical", "chance"]
    vals = [r["structured_mean"], r["classical_raw_mean"], r["chance_mean"]]
    cols = [PALETTE["cqfa"], PALETTE["classical_fa"], PALETTE["neutral"]]
    err = [[r["structured_mean"] - r["structured_ci_lo"], r["classical_raw_mean"] - r["classical_raw_ci_lo"], 0],
           [r["structured_ci_hi"] - r["structured_mean"], r["classical_raw_ci_hi"] - r["classical_raw_mean"], 0]]
    ax.bar(labels, vals, color=cols, yerr=err, capsize=5, error_kw=dict(elinewidth=2, capthick=2))
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.4, 1.0)
    bar_labels(ax, labels)
    bold_ticks(ax); thin_ticks(ax, ny=6)
    save(fig, "f5_dlog_separation")


# ---------------------------------------------------------------- F6: hardware noise
def fig_hardware():
    print("F6 hardware post-selection under noise")
    path = R("results/hardware/noise_study_summary.csv")
    if not _has(path):
        return
    df = pd.read_csv(path)
    depol = df[df["backend_label"].str.contains("depol", case=False, na=False)].copy()
    if depol.empty:
        print("  SKIP (no depol rows)"); return
    shots = sorted(depol["shots"].unique())[-1]
    depol = depol[depol["shots"] == shots].sort_values("noise_strength")
    apply_style()
    fig, ax = new_fig()
    x = depol["noise_strength"].to_numpy(dtype=float)
    ax.plot(x, depol["mean_abs_err_postsel"], marker="o", color=PALETTE["coherent"], label="post-selected")
    ax.fill_between(x, depol["ci_lo_postsel"], depol["ci_hi_postsel"], color=PALETTE["coherent"], alpha=0.15)
    ax.plot(x, depol["mean_abs_err_raw"], marker="s", color=PALETTE["single_copy"], label="raw")
    ax.set_xlabel("depolarizing $p$")
    ax.set_ylabel("readout error")
    ax.set_xscale("log")
    ax.legend(loc="center left")
    bold_ticks(ax); thin_ticks(ax)
    save(fig, "f6_hardware_noise")


# ---------------------------------------------------------------- F4: benchmark per task
def fig_benchmark():
    print("F4 benchmark: CQFA vs classical FA per task (, per-task)")
    path = R("results/aggregated/benchmark_master.csv")
    if not _has(path):
        return
    df = pd.read_csv(path)
    task_col = next((c for c in ["task", "task_name"] if c in df.columns), None)
    if task_col is None or "acc_mean" not in df.columns:
        print("  SKIP (schema)"); return
    # only the quantum models; grouped bars cqfa vs classical_fa vs non_equivariant per task
    tasks = ["coherent_teacher", "genomics_rc", "spt_derived", "orbit_interference", "symmetry_vs_broken"]
    tasks = [t for t in tasks if t in set(df[task_col])]
    short = {"coherent_teacher": "coh-teach", "genomics_rc": "genomics", "spt_derived": "SPT",
             "orbit_interference": "orbit", "symmetry_vs_broken": "sym/broken"}
    _ = (tasks, short)  # (full cross-task table lives in the supplement)
    # F4 = coherent-teacher across groups (CQFA's design regime): CQFA vs classical FA.
    d = df[df[task_col] == "coherent_teacher"]
    if d.empty:
        print("  SKIP (no coherent_teacher)"); return
    groups = ["z2_rc", "z2_pauli2", "z2_pauli3", "z2_pauli4", "c4", "c8", "d4"]
    glab = {"z2_rc": r"$\mathbb{Z}_2$", "z2_pauli2": r"$\mathbb{Z}_2^{2}$", "z2_pauli3": r"$\mathbb{Z}_2^{3}$",
            "z2_pauli4": r"$\mathbb{Z}_2^{4}$", "c4": r"$C_4$", "c8": r"$C_8$", "d4": r"$D_4$"}
    groups = [g for g in groups if g in set(d["rep"])]
    apply_style()
    fig, ax = new_fig(w=4.7, h=3.0)
    x = np.arange(len(groups)); w = 0.38
    for i, (m, lab) in enumerate([("cqfa", "CQFA"), ("classical_fa", "classical FA")]):
        vals = [d[(d["rep"] == g) & (d["model"] == m)]["acc_mean"].mean() for g in groups]
        ax.bar(x + (i - 0.5) * w, vals, w, color=PALETTE[m], label=lab)
    ax.axhline(0.5, color="k", lw=1.0, ls=":")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.5, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels([glab[g] for g in groups], fontweight="bold")
    ax.legend(loc="upper right", ncol=1)
    bold_ticks(ax); thin_ticks(ax, ny=6)
    save(fig, "f4_benchmark")


# ---------------------------------------------------------------- F2b: exponential separation
def fig_exponential_separation():
    print("F2b exponential measurement-complexity separation (full-chain string)")
    path = R("results/spt_measurement_cost/exponential_separation.csv")
    if not _has(path):
        return
    df = pd.read_csv(path).sort_values("n")
    apply_style()
    fig, ax = new_fig()
    ax.plot(df["n"], df["shots_shadow_singlecopy"], marker="s", color=PALETTE["single_copy"],
            label="single-copy (shadows)")
    ax.plot(df["n"], np.clip(df["shots_coherent"], 1.0, None), marker="o", color=PALETTE["coherent"],
            label="coherent readout")
    # theoretical 3^w guide
    ax.plot(df["n"], df["three_pow_w"] / 0.01, ls=":", color=PALETTE["neutral"], lw=1.8,
            label=r"$3^{w}/\varepsilon^2$")
    ax.set_xlabel(r"system size $n$")
    ax.set_ylabel(r"copies to $\varepsilon{=}0.1$")
    ax.set_yscale("log")
    ax.legend(loc="center left", fontsize=10, frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="0.75")
    bold_ticks(ax); thin_ticks(ax, nx=6)
    save(fig, "f2b_exponential_separation")


# ---------------------------------------------------------------- F8: trainability law
def fig_trainability_law():
    print("F8 trainability law: escape vs milder-plateau vs generic")
    path = R("results/trainability_law/trainability_law.csv")
    if not _has(path):
        return
    df = pd.read_csv(path)
    apply_style()
    fig, ax = new_fig(w=4.0, h=3.1)
    mp = df[df["experiment"] == "milder_plateau"].sort_values("n")
    # generic ansatz (steep exp decay) and CQFA on Z2 (milder exp decay)
    ax.plot(mp["n"], mp["var_generic"], marker="s", color=PALETTE["non_equivariant"],
            label="generic ansatz")
    ax.plot(mp["n"], mp["var_cqfa"], marker="o", color=PALETTE["classical_fa"],
            label=r"CQFA, $\mathbb{Z}_2$ (exp. $H_+$)")
    # escape families: constant dim(H_+) -> FLAT variance
    for dh, col, mk in [(4, PALETTE["cqfa"], "^"), (8, PALETTE["accent"], "D")]:
        e = df[(df["experiment"] == "escape") & (df["dim_hplus"] == dh)].sort_values("n")
        if not e.empty:
            ax.plot(e["n"], e["var_cqfa"], marker=mk, color=col,
                    label=fr"CQFA, large $G$ ($\dim H_+{{=}}{dh}$)")
    ax.set_xlabel(r"qubit count $n$")
    ax.set_ylabel(r"gradient variance")
    ax.set_yscale("log")
    ax.legend(loc="lower left", fontsize=9.5)
    bold_ticks(ax); thin_ticks(ax, nx=6)
    save(fig, "f8_trainability_law")


# ---------------------------------------------------------------- F9: coherent surplus
def fig_surplus():
    print("F9 coherent-surplus dimension (quadratic vs linear)")
    path = R("results/coherent_surplus/surplus_dimension.csv")
    if not _has(path):
        return
    df = pd.read_csv(path)
    cyc = df[df["name"].str.startswith("C") & (df["dim"] == df["order"])].sort_values("order")
    if cyc.empty:
        cyc = df.sort_values("order")
    apply_style()
    fig, ax = new_fig()
    ax.plot(cyc["order"], cyc["coherent_dim"], marker="o", color=PALETTE["cqfa"],
            label=r"coherent ($|G|^2$)")
    ax.plot(cyc["order"], cyc["classical_dim"], marker="s", color=PALETTE["classical_fa"],
            label=r"classical FA ($|G|$)")
    ax.plot(cyc["order"], cyc["surplus_dim"], marker="^", ls="--", color=PALETTE["accent"],
            label=r"surplus")
    ax.set_xlabel(r"group order $|G|$")
    ax.set_ylabel("feature-space dim.")
    ax.legend(loc="upper left", fontsize=10.5)
    bold_ticks(ax); thin_ticks(ax)
    save(fig, "f9_surplus_dimension")


# ---------------------------------------------------------------- F10: generalization
def fig_generalization():
    print("F10 generalization / sample efficiency")
    path = R("results/generalization/learning_curves.csv")
    if not _has(path):
        return
    df = pd.read_csv(path)
    apply_style()
    fig, ax = new_fig()
    st = {"cqfa": ("CQFA", PALETTE["cqfa"], "o"),
          "classical_fa": ("classical FA", PALETTE["classical_fa"], "s"),
          "non_equivariant": ("non-equivariant", PALETTE["non_equivariant"], "^")}
    for m, (lab, col, mk) in st.items():
        d = df[df["model"] == m].sort_values("n_train")
        if d.empty:
            continue
        ax.plot(d["n_train"], d["acc_mean"], marker=mk, color=col, label=lab)
        ax.fill_between(d["n_train"], d["ci_lo"], d["ci_hi"], color=col, alpha=0.15)
    ax.set_xlabel("training samples")
    ax.set_ylabel("test accuracy")
    ax.set_xscale("log")
    ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="0.75")
    bold_ticks(ax); thin_ticks(ax)
    save(fig, "f10_generalization")


# ---------------------------------------------------------------- F12: copy-permutation unification
def fig_copy_fusion():
    print("F12 copy-permutation unification (surplus over S_k = multi-copy advantage)")
    path = R("results/copy_fusion/copy_permutation_advantage.csv")
    if not _has(path):
        return
    df = pd.read_csv(path).sort_values("n")
    apply_style()
    fig, ax = new_fig()
    ax.plot(df["n"], df["copies_singlecopy"], marker="s", color=PALETTE["single_copy"],
            label="single-copy")
    ax.plot(df["n"], df["copies_coherent"], marker="o", color=PALETTE["coherent"],
            label=r"coherent $S_2$ (SWAP)")
    ax.set_xlabel(r"system size $n$")
    ax.set_ylabel(r"copies to $\varepsilon{=}0.1$")
    ax.set_yscale("log")
    ax.legend(loc="center left", fontsize=10.5)
    bold_ticks(ax); thin_ticks(ax, nx=6)
    save(fig, "f12_copy_fusion")


# ---------------------------------------------------------------- F11: two-knob independence
def fig_two_knob():
    print("F11 two-knob independence grid")
    path = R("results/trainability_law/two_knob_sweep.csv")
    if not _has(path):
        return
    df = pd.read_csv(path)
    apply_style()
    fig, ax = new_fig(w=4.0, h=3.1)
    cols = [PALETTE["cqfa"], PALETTE["classical_fa"], PALETTE["non_equivariant"], PALETTE["accent"]]
    for i, (gname, gdf) in enumerate(df.groupby("group")):
        ax.plot(gdf["trainability_reading"], gdf["hardness_reading"], marker="o",
                ls="-", color=cols[i % len(cols)], label=fr"$\dim H_+{{=}}{int(gdf['dim_hplus'].iloc[0])}$")
    ax.axhline(0.5, color="k", lw=1.0, ls=":")
    ax.set_xlabel("trainability reading")
    ax.set_ylabel("hardness (escape)")
    ax.set_xscale("log")
    ax.set_ylim(-0.1, 1.25)
    ax.legend(loc="upper center", ncol=2, fontsize=9, frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="0.75")
    bold_ticks(ax); thin_ticks(ax)
    save(fig, "f11_two_knob")


def main():
    for f in (fig_measurement_complexity, fig_exponential_separation, fig_gradient_variance,
              fig_dial, fig_dla, fig_dlog, fig_hardware, fig_benchmark, fig_trainability_law,
              fig_surplus, fig_generalization, fig_two_knob, fig_copy_fusion):
        try:
            f()
        except Exception as e:
            print(f"  ERROR in {f.__name__}: {type(e).__name__}: {e}")
    print("done. figures in figures/output/")


if __name__ == "__main__":
    main()
