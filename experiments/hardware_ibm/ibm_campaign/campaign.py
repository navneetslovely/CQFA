
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .specs import build_campaign
from . import runner as R
from . import analyze as A
from .._util import package_versions, _git_hash


ALL = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")
DEFAULT_ROOT = "results/ibm_campaign"


# ---------------------------------------------------------------------------
# JSON helpers + token scrub
# ---------------------------------------------------------------------------
def _jsonable(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return _jsonable(obj.tolist())
    return obj


def _scrub_token(text: str) -> str:
    """Defensive: never let a token value leak into an artifact."""
    tok = os.environ.get("QISKIT_IBM_TOKEN")
    if tok and tok in text:
        raise RuntimeError("refusing to write artifact: API token found in payload")
    return text


def _dump_json(path, obj):
    text = json.dumps(_jsonable(obj), indent=2)
    _scrub_token(text)
    with open(path, "w") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def cmd_plan(specs, args):
    cost = R.estimate_cost(specs)
    print(f"\nCQFA IBM campaign plan  ({len(specs)} PUBs, "
          f"{cost['total_shots']} shots)\n" + "-" * 60)
    print(f"{'exp':<5}{'PUBs':>6}{'shots':>10}{'est_qpu_s':>12}")
    for exp in sorted(cost["by_experiment"]):
        e = cost["by_experiment"][exp]
        print(f"{exp:<5}{e['n_pubs']:>6}{e['shots']:>10}"
              f"{R.PER_SHOT_SECONDS * e['shots']:>12.2f}")
    print("-" * 60)
    print(f"{'TOTAL':<5}{cost['n_pubs']:>6}{cost['total_shots']:>10}"
          f"{cost['est_qpu_seconds']:>12.2f}")
    print(f"\nestimate = {R.JOB_OVERHEAD_SECONDS}s overhead + "
          f"{R.PER_SHOT_SECONDS}s/shot x {cost['total_shots']} shots "
          f"= {cost['est_qpu_seconds']}s (one batched job)")
    print(f"budget   = {args.budget_seconds}s  ->  "
          f"{'FITS' if cost['est_qpu_seconds'] <= args.budget_seconds else 'OVER BUDGET'}")
    return cost


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def _save_artifacts(out_dir, specs, results, args, which):
    os.makedirs(out_dir, exist_ok=True)
    # raw counts
    _dump_json(os.path.join(out_dir, "raw_counts.json"), results["counts_by_id"])
    # per-PUB index (experiment / shots / theory / meta / transpiled depth / layout)
    depth = results["transpiled_2q_depth_by_id"]
    layout = results.get("final_layout_by_id", {})
    pub_index = [{
        "pub_id": s.pub_id, "experiment": s.experiment, "analysis": s.analysis,
        "shots": s.shots, "theory": s.theory, "meta": s.meta,
        "transpiled_2q_depth": depth.get(s.pub_id),
        "final_index_layout": layout.get(s.pub_id),
    } for s in specs]
    _dump_json(os.path.join(out_dir, "pub_index.json"), pub_index)
    # circuits: logical (device-independent) + transpiled (the physical circuits that
    # actually ran). Best-effort — raw_counts/manifest are already on disk, so a qpy
    # hiccup must not discard a completed (expensive) device run.
    try:
        from qiskit import qpy
        with open(os.path.join(out_dir, "circuits.qpy"), "wb") as f:
            qpy.dump([s.circuit for s in specs], f)
        tcircs = results.get("transpiled_circuits")
        if tcircs:
            with open(os.path.join(out_dir, "circuits_transpiled.qpy"), "wb") as f:
                qpy.dump(list(tcircs), f)
    except Exception as e:
        print(f"[warn] qpy dump skipped: {e}")
    # manifest (backend / job id / usage / calibration / versions / cost)
    manifest = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_hash": _git_hash(), "argv": list(sys.argv),
        "which_experiments": list(which),
        "backend_label": results["backend_label"], "job_id": results["job_id"],
        "usage": results["usage"], "calibration": results["calibration"],
        "cost_estimate": results["cost_estimate"], "versions": package_versions(),
        "n_pubs": len(specs), "note": "API token intentionally NOT stored",
    }
    _dump_json(os.path.join(out_dir, "manifest.json"), manifest)
    return pub_index


def _write_csvs(out_dir, analyzed):
    from .._util import write_csv
    adir = os.path.join(out_dir, "analysis")
    os.makedirs(adir, exist_ok=True)
    paths = []
    for tag, rows in analyzed.items():
        if not rows:
            continue
        p = os.path.join(adir, f"{tag.lower()}.csv")
        write_csv(p, rows, list(rows[0].keys()))
        paths.append(p)
    return paths


def _dry_run_gate(analyzed):
    """On an ideal-sampler run, assert every estimate matches theory within a
    generous shot tolerance. Returns (ok, report_lines)."""
    import numpy as np
    ok = True
    lines = []

    def tol(nshots, floor=0.05):
        return max(floor, 6.0 / np.sqrt(max(1, nshots)))

    for r in analyzed.get("C1", []):
        t = tol(r["shots"])
        good = r["abs_err"] <= t
        ok &= good
        lines.append(f"  C1 {r['pub_id']:<16} err={r['abs_err']:.3f} tol={t:.3f} "
                     f"{'ok' if good else 'FAIL'}")
    for r in analyzed.get("C2", []):
        t = tol(r["shots"])
        good = r["abs_err"] <= t
        ok &= good
        lines.append(f"  C2 {r['pub_id']:<16} err={r['abs_err']:.3f} tol={t:.3f} "
                     f"{'ok' if good else 'FAIL'}")
    for exp in ("C3", "C6", "C7"):
        for r in analyzed.get(exp, []):
            t = tol(r.get("n_accepted", r["shots"]), floor=0.08)
            good = (np.isnan(r["abs_err_postsel"]) or r["abs_err_postsel"] <= t)
            ok &= good
            lines.append(f"  {exp} {r['pub_id']:<16} err_ps={r['abs_err_postsel']:.3f} "
                         f"tol={t:.3f} acc={r['accept_rate']:.2f} "
                         f"{'ok' if good else 'FAIL'}")
    for r in analyzed.get("C4", []):
        good = np.isnan(r["abs_err_surplus"]) or r["abs_err_surplus"] <= 0.10
        ok &= good
        lines.append(f"  C4 {r['input_id']:<12} surplus_hat={r['surplus_hat']:.3f} "
                     f"exact={r['surplus_exact']:.3f} err={r['abs_err_surplus']:.3f} "
                     f"{'ok' if good else 'FAIL'}")
    for r in analyzed.get("C5", []):
        good = r["equivariance_gap_vs_g0"] <= tol(r["shots"], floor=0.12)
        ok &= good
        lines.append(f"  C5 {r['input_id']}/g{r['g']} gap={r['equivariance_gap_vs_g0']:.3f} "
                     f"{'ok' if good else 'FAIL'}")
    for r in analyzed.get("C8", []):
        t = tol(r["shots"], floor=0.08)
        if r["mode"] == "F":
            good = (r["success"] == 1) and (r["abs_err"] <= t)   # peak at k=s, prob~1
        else:
            good = (r["abs_err"] <= t)                            # Mode B uniform (TV~0)
        ok &= good
        lines.append(f"  C8 {r['pub_id']:<16} mode={r['mode']} peak_k={r['peak_k']} "
                     f"s={r['hidden_shift']} peak_p={r['peak_prob']:.3f} "
                     f"tv={r['tv_from_uniform']:.3f} {'ok' if good else 'FAIL'}")
    for r in analyzed.get("C1_shadow", []):
        lines.append(f"  C1_shadow n={r['n']} hits={r['n_hits']}/{r['n_snapshots']} "
                     f"rate={r['hit_rate']:.3f} exp={r['expected_hit_rate']:.4f} "
                     f"3^w={r['three_pow_w']}")
    return ok, lines


def cmd_run(specs, args, which):
    tag = args.tag or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = os.path.join(args.out, tag)
    print(cmd_plan(specs, args) and "")

    if args.backend == "ibm":
        results = R.run_ibm(specs, args)
    else:
        results = R.run_offline(specs, backend_kind=args.backend, seed=args.seed)

    pub_index = _save_artifacts(out_dir, specs, results, args, which)
    analyzed = A.analyze_all(specs, results["counts_by_id"])
    paths = _write_csvs(out_dir, analyzed)

    print(f"\nbackend={results['backend_label']} job_id={results['job_id']} "
          f"-> {out_dir}")
    print(f"wrote {len(paths)} analysis CSVs, {len(pub_index)} PUBs, "
          f"raw_counts.json, manifest.json, circuits.qpy")

    if args.backend == "aer_noiseless":
        ok, lines = _dry_run_gate(analyzed)
        print("\nCORRECTNESS GATE (ideal sampler vs exact theory):")
        print("\n".join(lines))
        print("\nGATE:", "PASS — estimates match theory" if ok
              else "FAIL — see rows above")
        if not ok:
            sys.exit(2)
    return out_dir


# ---------------------------------------------------------------------------
# analyze (re-analyze saved raw counts, no execution)
# ---------------------------------------------------------------------------
def cmd_analyze(args, which):
    out_dir = os.path.join(args.out, args.tag)
    with open(os.path.join(out_dir, "raw_counts.json")) as f:
        counts_by_id = json.load(f)
    # deterministic rebuild of specs (theory + meta) — identical to the run
    specs = build_campaign(which=which)
    analyzed = A.analyze_all(specs, counts_by_id)
    paths = _write_csvs(out_dir, analyzed)
    print(f"re-analyzed {args.tag}: wrote {len(paths)} CSVs to {out_dir}/analysis")
    return analyzed


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="CQFA IBM hardware campaign (7 experiments)")
    ap.add_argument("command", choices=["plan", "run", "analyze"])
    ap.add_argument("--backend", default="aer_noiseless",
                    help="aer_noiseless | fake:fez | fake:marrakesh | ibm")
    ap.add_argument("--which", default=",".join(ALL),
                    help="comma list of experiments, e.g. C1,C3,C4")
    ap.add_argument("--tag", default=None, help="run tag / subdir")
    ap.add_argument("--out", default=DEFAULT_ROOT)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--budget-seconds", type=float, default=300.0)
    ap.add_argument("--ibm", action="store_true", help="lock 2/3 for the real device")
    ap.add_argument("--ibm-confirm", action="store_true", help="lock 3/3 for the real device")
    ap.add_argument("--max-shots", type=int, default=8192)
    ap.add_argument("--max-circuits", type=int, default=400)
    ap.add_argument("--max-total-shots", type=int, default=1_000_000)
    args = ap.parse_args(argv)

    which = tuple(w.strip() for w in args.which.split(",") if w.strip())
    specs = build_campaign(which=which)

    if args.command == "plan":
        cmd_plan(specs, args)
    elif args.command == "run":
        cmd_run(specs, args, which)
    elif args.command == "analyze":
        cmd_analyze(args, which)


if __name__ == "__main__":
    main()
