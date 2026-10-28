"""
experiments.multitask_benchmark.run_benchmarks
====================================
The single driver for the CQFA multi-dataset benchmark sweep.

Runs the full (group x task x model) matrix (see matrix.py) via
cqfa.train.run_experiment, aggregates over >=5 seeds with bootstrap 95% CIs, and
writes ONE tidy master CSV to results/aggregated/benchmark_master.csv plus a
self-describing per-cell run folder under runs/benchmarks/<cell>/ for every
model-run (traceability: every headline number -> its source CSVs).

Determinism contract
--------------------
  * data seed = 1000 + seed (matches cqfa.train.run_experiment);
  * teacher observable fixed at teacher_seed=12345 (concept fixed across seeds);
  * genomics-RC uses the always-available synthetic RC-invariant generator
    (fully offline, no network); dataset_source recorded per row.

Scope of each row
-----------------
Each row carries advantage_claim in
{expressivity, sample_complexity, inductive_bias, mechanism_only} so no
downstream table over-reads a number. quantum_model=False flags the classical
RC baseline. NO runtime-supremacy claim at this simulable scale.

CLI
---
    python -m experiments.multitask_benchmark.run_benchmarks --seeds 0 1 2 3 4
    python -m experiments.multitask_benchmark.run_benchmarks --quick        # smoke: 1 seed, tiny
Filters: --cells B1 B2 ... , --only-rep z2_rc , --only-task orbit_interference .
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time

import numpy as np

from cqfa import train as Tr
from cqfa import tasks as T
from datasets import genomics_reverse_complement as GR
from datasets import rc_symmetric_baseline as RCB
from . import metrics as MET
from .matrix import CELLS, resolve_rep

# repo root = two levels up from this file (experiments/multitask_benchmark/ -> repo)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# master CSV schema (order is primary: test_master_csv_schema pins it)
MASTER_FIELDS = [
    "cell", "rep", "task", "model", "advantage_claim", "dataset_source",
    "n_seeds", "n_params", "n_qubits", "group_order", "rep_dim",
    "acc_mean", "acc_std", "acc_ci_lo", "acc_ci_hi",
    "balacc_mean", "balacc_std", "balacc_ci_lo", "balacc_ci_hi",
    "mcc_mean", "mcc_std", "mcc_ci_lo", "mcc_ci_hi",
    "auprc_mean", "auprc_std", "auprc_ci_lo", "auprc_ci_hi",
    "class_balance", "n_train", "n_test", "epochs", "lr", "n_layers",
    "teacher_seed", "quantum_model", "status", "git_commit", "timestamp",
    "run_folder",
]


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "nogit"
    except Exception:
        return "nogit"


def _n_qubits(dim):
    n = int(round(np.log2(dim)))
    return n if 2 ** n == dim else -1


def _quantum_row(cell_id, rep_key, task_name, model_name, claim, dataset_id,
                 seeds, cfg, teacher_seed, git_commit, runs_root):
    """Run one (cell, rep, model) quantum-model cell -> one master-CSV row dict."""
    # resolve the representation (genomics uses a bespoke U_RC rep + task closure)
    if dataset_id == "genomics_rc":
        task_fn = GR.make_task(k=cfg.get("k", 2), seq_len=cfg.get("seq_len", 12),
                               motif=cfg.get("motif", "CG"), rho=cfg.get("rho", 0.1))
        rep = task_fn.rep
        rep_name = rep.name
    else:
        rep = resolve_rep(rep_key)
        rep_name = rep_key
        task_fn = T.make_task(task_name, rep, teacher_seed=teacher_seed)

    dim = rep.dim
    nq = _n_qubits(dim)
    if nq < 0:                                   # power-of-two guard (defensive)
        return dict(cell=cell_id, rep=rep_name, task=task_name, model=model_name,
                    advantage_claim=claim, dataset_source="n/a", n_seeds=0,
                    n_params="", n_qubits="", group_order=rep.order, rep_dim=dim,
                    class_balance="", n_train=cfg["n_train"], n_test=cfg["n_test"],
                    epochs=cfg["epochs"], lr=cfg["lr"], n_layers=cfg["n_layers"],
                    teacher_seed=teacher_seed, quantum_model=True,
                    status="skipped_nonpow2", git_commit=git_commit,
                    timestamp=time.strftime("%Y%m%d-%H%M%S"), run_folder="",
                    **{f: "" for f in MASTER_FIELDS
                       if f.endswith(("_mean", "_std", "_ci_lo", "_ci_hi"))})

    out_root = os.path.join(runs_root, cell_id)
    summary, folder, pred_rows = Tr.run_experiment(
        rep_name, model_name, seeds,
        n_train=cfg["n_train"], n_test=cfg["n_test"], epochs=cfg["epochs"],
        lr=cfg["lr"], n_layers=cfg["n_layers"], tag=cell_id, out_root=out_root,
        teacher_seed=teacher_seed, write=True, task_fn=task_fn,
        task_name=task_name, rep=rep, return_preds=True)

    agg = MET.aggregate(pred_rows, ci_seed=teacher_seed)
    dataset_source = "spt_states" if dataset_id == "spt_states" else (
        "synthetic" if dataset_id == "synthetic" else "synthetic")  # genomics synth
    run_folder = os.path.relpath(folder, REPO_ROOT) if folder else ""
    return dict(
        cell=cell_id, rep=rep_name, task=task_name, model=model_name,
        advantage_claim=claim, dataset_source=dataset_source,
        n_seeds=len(seeds), n_params=summary["n_params"], n_qubits=nq,
        group_order=rep.order, rep_dim=dim,
        acc_mean=agg["acc_mean"], acc_std=agg["acc_std"],
        acc_ci_lo=agg["acc_ci_lo"], acc_ci_hi=agg["acc_ci_hi"],
        balacc_mean=agg["balacc_mean"], balacc_std=agg["balacc_std"],
        balacc_ci_lo=agg["balacc_ci_lo"], balacc_ci_hi=agg["balacc_ci_hi"],
        mcc_mean=agg["mcc_mean"], mcc_std=agg["mcc_std"],
        mcc_ci_lo=agg["mcc_ci_lo"], mcc_ci_hi=agg["mcc_ci_hi"],
        auprc_mean=agg["auprc_mean"], auprc_std=agg["auprc_std"],
        auprc_ci_lo=agg["auprc_ci_lo"], auprc_ci_hi=agg["auprc_ci_hi"],
        class_balance=summary["class_balance"], n_train=cfg["n_train"],
        n_test=cfg["n_test"], epochs=cfg["epochs"], lr=cfg["lr"],
        n_layers=cfg["n_layers"], teacher_seed=teacher_seed, quantum_model=True,
        status="ok", git_commit=git_commit,
        timestamp=time.strftime("%Y%m%d-%H%M%S"), run_folder=run_folder)


def _classical_rc_row(cell_id, claim, seeds, cfg, git_commit):
    """Run the classical RC-equivariant baseline over the same seeds -> one row."""
    k = cfg.get("k", 2)
    pred_rows, accs, nparams = [], [], []
    for s in seeds:
        r = RCB.run_rc_baseline_seed(s, cfg["n_train"], cfg["n_test"], k=k,
                                     seq_len=cfg.get("seq_len", 12),
                                     motif=cfg.get("motif", "CG"),
                                     rho=cfg.get("rho", 0.1))
        pred_rows.append(r); accs.append(r["acc"]); nparams.append(r["n_params"])
    agg = MET.aggregate(pred_rows, ci_seed=12345)
    return dict(
        cell=cell_id, rep=f"RC-k{k}", task="genomics_rc", model="rc_cnn_classical",
        advantage_claim=claim, dataset_source="synthetic", n_seeds=len(seeds),
        n_params=int(nparams[0]), n_qubits=int(2 * k), group_order=2, rep_dim=4 ** k,
        acc_mean=agg["acc_mean"], acc_std=agg["acc_std"],
        acc_ci_lo=agg["acc_ci_lo"], acc_ci_hi=agg["acc_ci_hi"],
        balacc_mean=agg["balacc_mean"], balacc_std=agg["balacc_std"],
        balacc_ci_lo=agg["balacc_ci_lo"], balacc_ci_hi=agg["balacc_ci_hi"],
        mcc_mean=agg["mcc_mean"], mcc_std=agg["mcc_std"],
        mcc_ci_lo=agg["mcc_ci_lo"], mcc_ci_hi=agg["mcc_ci_hi"],
        auprc_mean=agg["auprc_mean"], auprc_std=agg["auprc_std"],
        auprc_ci_lo=agg["auprc_ci_lo"], auprc_ci_hi=agg["auprc_ci_hi"],
        class_balance=float(np.mean([r["y_true"].mean() for r in pred_rows])),
        n_train=cfg["n_train"], n_test=cfg["n_test"], epochs=cfg["epochs"],
        lr=cfg["lr"], n_layers=cfg["n_layers"], teacher_seed=12345,
        quantum_model=False, status="ok", git_commit=git_commit,
        timestamp=time.strftime("%Y%m%d-%H%M%S"), run_folder="")


def _write_master(path, rows):
    """Write the master CSV atomically (tmp + os.replace)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MASTER_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


def run(seeds, cells_filter=None, only_rep=None, only_task=None, cfg=None,
        out_csv=None, runs_root=None, teacher_seed=12345, verbose=True):
    """Execute the matrix and write the master CSV. Returns (rows, out_csv)."""
    cfg = cfg or dict(n_train=80, n_test=60, epochs=120, lr=0.15, n_layers=3)
    out_csv = out_csv or os.path.join(REPO_ROOT, "results", "aggregated",
                                      "benchmark_master.csv")
    runs_root = runs_root or os.path.join(REPO_ROOT, "runs", "benchmarks")
    git_commit = _git_commit()
    rows = []

    for (cell_id, rep_keys, task_name, model_names, claim, dataset_id, extra) in CELLS:
        if cells_filter and cell_id not in cells_filter:
            continue
        if only_task and task_name != only_task:
            continue
        cell_cfg = dict(cfg); cell_cfg.update(extra)
        for rep_key in rep_keys:
            if only_rep and rep_key != only_rep and dataset_id != "genomics_rc":
                continue
            for model_name in model_names:
                if model_name == "rc_cnn_classical":
                    row = _classical_rc_row(cell_id, claim, seeds, cell_cfg, git_commit)
                else:
                    row = _quantum_row(cell_id, rep_key, task_name, model_name, claim,
                                       dataset_id, seeds, cell_cfg, teacher_seed,
                                       git_commit, runs_root)
                rows.append(row)
                if verbose:
                    am = row.get("acc_mean", "")
                    am = f"{am:.3f}" if isinstance(am, float) else str(am)
                    print(f"  [{cell_id}] {row['rep']:<10} {task_name:<18} "
                          f"{row['model']:<16} acc={am:<6} {row['status']}")
    _write_master(out_csv, rows)
    if verbose:
        _print_summary(rows, out_csv)
    return rows, out_csv


def _print_summary(rows, out_csv):
    print("\n" + "=" * 72)
    print("BENCHMARK SUMMARY (cqfa vs classical_fa mean accuracy, with claim)")
    print("=" * 72)
    by_cell = {}
    for r in rows:
        if r.get("status") != "ok" or not isinstance(r.get("acc_mean"), float):
            continue
        by_cell.setdefault((r["cell"], r["rep"], r["task"], r["advantage_claim"]), {})[
            r["model"]] = r["acc_mean"]
    print(f"{'cell':<5}{'rep':<11}{'task':<18}{'cqfa':>7}{'clFA':>7}{'gap':>7}  claim")
    print("-" * 72)
    for (cell, rep, task, claim), md in by_cell.items():
        cq = md.get("cqfa"); fa = md.get("classical_fa")
        gap = (cq - fa) if (cq is not None and fa is not None) else float("nan")
        cqs = f"{cq:.3f}" if cq is not None else "  -  "
        fas = f"{fa:.3f}" if fa is not None else "  -  "
        gaps = f"{gap:+.3f}" if gap == gap else "  -  "
        print(f"{cell:<5}{rep:<11}{task:<18}{cqs:>7}{fas:>7}{gaps:>7}  {claim}")
    print("-" * 72)
    print(f"master CSV -> {out_csv}")
    print("NOTE: advantage_claim gates the interpretation; genomics-RC is the "
          "classical-competitive anchor (no quantum-advantage claim).")


def main(argv=None):
    ap = argparse.ArgumentParser(description="CQFA multi-dataset benchmark sweep.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--cells", nargs="+", default=None,
                    help="subset of cell ids (default: all, e.g. B1 B2)")
    ap.add_argument("--only-rep", default=None)
    ap.add_argument("--only-task", default=None)
    ap.add_argument("--out", default=None, help="master CSV path")
    ap.add_argument("--runs-root", default=None, help="per-cell run-folder root")
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.15)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--teacher-seed", type=int, default=12345)
    ap.add_argument("--quick", action="store_true",
                    help="smoke: 1 seed, tiny sizes, all cells")
    args = ap.parse_args(argv)

    if args.quick:
        seeds = [0]
        cfg = dict(n_train=16, n_test=16, epochs=15, lr=0.2, n_layers=2)
    else:
        seeds = args.seeds
        cfg = dict(n_train=args.n_train, n_test=args.n_test, epochs=args.epochs,
                   lr=args.lr, n_layers=args.n_layers)

    print(f"CQFA benchmark sweep | seeds={seeds} | cfg={cfg}")
    rows, out_csv = run(seeds, cells_filter=args.cells, only_rep=args.only_rep,
                        only_task=args.only_task, cfg=cfg, out_csv=args.out,
                        runs_root=args.runs_root, teacher_seed=args.teacher_seed)
    return rows, out_csv


if __name__ == "__main__":
    main()
