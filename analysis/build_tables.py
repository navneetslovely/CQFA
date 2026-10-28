"""Generate the result tables directly from the recorded CSVs.

No value is written by hand: every table cell is read from a file under
results/. Output is emitted as LaTeX fragments.
"""
from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd

from cqfa.paths import ROOT as _ARCHIVE_ROOT

ROOT = str(_ARCHIVE_ROOT)
R = lambda *p: os.path.join(ROOT, *p)
OUT = R("analysis", "output", "tables.tex")
_parts = []


def emit(s): _parts.append(s)


def _fmt(x, nd=3):
    try:
        if x != x or x in (np.inf, -np.inf):
            return r"$\infty$" if x == np.inf else "--"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


# ---- exponential measurement-complexity separation ----------------------
def tab_e2():
    p = R("results/spt_measurement_cost/exponential_separation.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p).sort_values("n")
    emit(r"\begin{table}[t]\centering\small")
    emit(r"\caption{Exponential measurement-complexity separation on SPT data: copies to "
         r"estimate the weight-$w$ ($w{=}n{-}2$) nonlocal string order to $\varepsilon{=}0.1$. "
         r"The single-copy shadow variance equals $3^{w}$ (verified $V_{\mathrm{shadow}}/3^{w}"
         r"\!\approx\!1$), so its cost is exponential; the coherent readout is $O(1)$. "
         r"\emph{Source:} \texttt{results/spt\_measurement\_cost/exponential\_separation.csv}.}")
    emit(r"\label{tab:e2}")
    emit(r"\begin{tabular}{rrrrr}")
    emit(r"\toprule")
    emit(r"$n$ & $w$ & $V_{\mathrm{shadow}}/3^{w}$ & single-copy & coherent \\")
    emit(r"\midrule")
    for _, r in d.iterrows():
        emit(f"{int(r['n'])} & {int(r['weight'])} & {_fmt(r['V_shadow_over_3w'],2)} & "
             f"{_fmt(r['shots_shadow_singlecopy'],0)} & {_fmt(max(r['shots_coherent'],1),0)} " + r"\\")
    emit(r"\bottomrule\end{tabular}\end{table}")


# ---- Full benchmark (task x model, mean acc) --------------------------------
def tab_benchmark():
    p = R("results/aggregated/benchmark_master.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    tcol = "task" if "task" in d.columns else "task_name"
    tasks = [t for t in ["coherent_teacher", "genomics_rc", "spt_derived",
                         "orbit_interference", "symmetry_vs_broken"] if t in set(d[tcol])]
    models = [m for m in ["cqfa", "classical_fa", "non_equivariant", "augmented"] if m in set(d["model"])]
    mlab = {"cqfa": "CQFA", "classical_fa": "cl-FA", "non_equivariant": "non-eq", "augmented": "aug"}
    tlab = {"coherent_teacher": "coherent-teacher", "genomics_rc": "genomics (RC)",
            "spt_derived": "SPT-derived", "orbit_interference": "orbit-interference",
            "symmetry_vs_broken": "symmetry-vs-broken"}
    emit(r"\begin{table}[t]\centering\small")
    emit(r"\caption{Multi-dataset benchmark: mean test accuracy over 5 seeds, averaged "
         r"across all groups per task. CQFA leads where the discriminative signal is a "
         r"coherent quantity (coherent-teacher) and is competitive on genomics; on "
         r"symmetry-broken-signal tasks output-averaging suffices, so CQFA is (by design) "
         r"not the right bias---a characterized boundary, not a failure. \emph{Source:} "
         r"\texttt{results/aggregated/benchmark\_master.csv}.}")
    emit(r"\label{tab:bench}")
    emit(r"\begin{tabular}{l" + "r" * len(models) + "}")
    emit(r"\toprule")
    emit("Task & " + " & ".join(mlab[m] for m in models) + r" \\")
    emit(r"\midrule")
    for t in tasks:
        row = [tlab[t]]
        best = max(d[(d[tcol] == t) & (d["model"] == m)]["acc_mean"].mean() for m in models)
        for m in models:
            v = d[(d[tcol] == t) & (d["model"] == m)]["acc_mean"].mean()
            cell = _fmt(v, 3)
            if abs(v - best) < 1e-9:
                cell = r"\textbf{" + cell + "}"
            row.append(cell)
        emit(" & ".join(row) + r" \\")
    emit(r"\bottomrule\end{tabular}\end{table}")


# ---- Hardware noise ---------------------------------------------------------
def tab_hardware():
    p = R("results/hardware/noise_study_summary.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    d = d[d["shots"] == d["shots"].max()]
    emit(r"\begin{table}[t]\centering\small")
    emit(r"\caption{Symmetry-protected post-selection under noise (highest shot budget, "
         r"mean over 5 seeds). Post-selection suppresses the readout error by up to two "
         r"orders of magnitude; the raw error is noise-independent. \emph{Source:} "
         r"\texttt{results/hardware/noise\_study\_summary.csv}.}")
    emit(r"\label{tab:hw}")
    emit(r"\begin{tabular}{lrr}")
    emit(r"\toprule")
    emit(r"Backend & post-selected err. & raw err. \\")
    emit(r"\midrule")
    order = ["Aer:noiseless", "Aer:depol(p=0.001)", "Aer:depol(p=0.01)",
             "Aer:depol(p=0.03)", "Aer:depol(p=0.1)", "Fake:manila"]
    for b in order:
        r_ = d[d["backend_label"] == b]
        if r_.empty:
            continue
        emit(f"{b.replace('_','')} & {_fmt(float(r_['mean_abs_err_postsel'].iloc[0]),3)} & "
             f"{_fmt(float(r_['mean_abs_err_raw'].iloc[0]),3)} " + r"\\")
    emit(r"\bottomrule\end{tabular}\end{table}")


# ---- DLA dimension ----------------------------------------------------------
def tab_dla():
    p = R("results/trainability_law/dla.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    gen = d[d["which"] == "generic_hea"].sort_values("n")
    cq = d[d["which"] == "cqfa_restricted"].sort_values("n")
    ns = sorted(d["n"].unique())
    emit(r"\begin{table}[t]\centering\small")
    emit(r"\caption{Dynamical-Lie-algebra dimension vs.\ qubit count. CQFA's frame-induced "
         r"algebra is polynomial ($3(n{-}1)$); the generic ansatz's is exponential "
         r"($4^n{-}1$). By the Lie-algebraic barren-plateau bound this is the structural "
         r"guarantee of trainability. \emph{Source:} \texttt{results/trainability\_law/dla.csv}.}")
    emit(r"\label{tab:dla}")
    emit(r"\begin{tabular}{l" + "r" * len(ns) + "}")
    emit(r"\toprule")
    emit(r"$\dim\mathfrak{g}$ & " + " & ".join(fr"$n{{=}}{int(n)}$" for n in ns) + r" \\")
    emit(r"\midrule")
    emit("CQFA (restricted) & " + " & ".join(
        _fmt(float(cq[cq['n']==n]['dim_g_restricted'].iloc[0]),0) if len(cq[cq['n']==n]) else "--" for n in ns) + r" \\")
    emit("Generic (ambient) & " + " & ".join(
        _fmt(float(gen[gen['n']==n]['dim_g_ambient'].iloc[0]),0) if len(gen[gen['n']==n]) else "--" for n in ns) + r" \\")
    emit(r"\bottomrule\end{tabular}\end{table}")


def main():
    emit("% Auto-generated by figures/scripts/make_tables.py from results/*.csv. Do not edit by hand.")
    for f in (tab_e2, tab_benchmark, tab_hardware, tab_dla):
        try:
            f()
        except Exception as e:
            emit(f"% ERROR {f.__name__}: {e}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(_parts) + "\n")
    print("wrote", os.path.relpath(OUT, ROOT), f"({len(_parts)} lines)")


if __name__ == "__main__":
    main()
