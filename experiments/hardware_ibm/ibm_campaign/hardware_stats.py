"""
experiments.hardware_ibm.ibm_campaign.hardware_stats
=================================================
Per-job and per-PUB hardware statistics for the recorded IBM campaign.

Builds the complete statistics table from artifacts already on disk under
``results/ibm_campaign/``, so the hardware interpretation rests only on what was
measured. NO NEW QPU RUNS. Nothing is deleted or overwritten; every output is a
new file.

WHAT THIS MODULE IS
-------------------
This item is DESCRIPTIVE. There is no hypothesis to confirm or kill, so
``criterion_outcome = DESCRIPTIVE_NO_CRITERION``. The deliverable is (a) a
complete statistics table with uncertainty on every reported quantity, (b) a
finite-N fit of the Mode-F information quantity with uncertainty bands over the N
values ACTUALLY MEASURED, and (c) an explicit audit of which fields were never
recorded.

PRE-REGISTERED ESTIMATORS AND RULES (fixed BEFORE the first run of this file)
-----------------------------------------------------------------------------
P1. Point estimates are RECOMPUTED from ``raw_counts.json`` and must reproduce the
    already-stored ``analysis/*.csv`` value to <= 1e-12. Column
    ``matches_stored_csv`` records this per row; any FALSE is reported as a defect,
    not silently smoothed.
P2. Binomial proportions (accept_rate, peak probability, shadow hit rate) get a
    95% WILSON score interval (continuity-uncorrected).
P3. Every other reported scalar gets a nonparametric MULTINOMIAL BOOTSTRAP over the
    stored shot counts: B = 10000 resamples, seed = 20260722, percentile 2.5/97.5.
    Both are fixed here before any number is read.
P4. DATA-SELECTED vs PRE-REGISTERED peak. ``peak_prob`` in the existing C8 CSV is
    P(argmax), i.e. the bin is chosen by the data, so its CI is optimistically
    biased. We report BOTH: ``p_at_true_shift`` (pre-registered bin k = s; this is
    the one) and ``p_at_argmax`` (data-selected, labelled).
P5. INFORMATION QUANTITY (the thing fitted vs N). For a frame-register outcome
    distribution p(k | s) over N = 2^n_frame bins measured at a known hidden shift s:
        R(N, s) = log2(N) - H(K | s)     [bits recovered per shot about the frame index]
    H is Shannon entropy in bits. Ideal Mode F: p = delta_s  =>  H = 0  =>  R = log2 N.
    Ideal Mode B: p uniform  =>  H = log2 N  =>  R = 0. Plug-in H is negatively
    biased, so R is positively biased; we therefore report BOTH the plug-in R and
    the MILLER-MADOW corrected R_mm (H_mm = H_plugin + (K_obs - 1) / (2 n ln 2)),
    and FIT THE CORRECTED ONE. We additionally report the pooled mutual information
    I(S;K) per (run, mode, N), which is capped at log2(n_shifts_measured) by the
    number of shifts actually run (3, 3, 2) -- a design limit, disclosed, not hidden.
P6. FIT. Weighted least squares of R_mm on log2(N) across N in {4, 8, 16} ONLY,
    weights 1/se^2, se from the per-arm bootstrap. Slope/intercept uncertainty by
    refitting each of the B joint bootstrap replicates (percentile 2.5/97.5).
    The fit is DESCRIPTIVE and LOCAL: 3 distinct N, 2-4 frame qubits. Three points
    cannot discriminate log2 N from any other function agreeing on them, and this
    module makes no asymptotic statement.
P7. VACUITY GUARD. ``label_signal_scale`` = |ideal_peak_prob - 1/N| for C8 rows,
    |exact value| for the others. Any cell < 1e-10 is marked
    ``vacuous_excluded`` BEFORE any interval is interpreted, and is excluded from
    the fit. Mode-B rows have signal scale exactly 0 BY CONSTRUCTION (they are the
    designed blind null), so they are tagged ``null_control_by_design`` and are
    reported as the null, never as a measurement of an effect.
P8. BASELINE CHOICE. R for hardware is differenced against, in order:
      weak baseline   = blind/uniform readout, R = 0 (chance) -- the flattering one;
      strong baseline = ideal noiseless simulation ``dry_c8_ideal``;
      strongest realistic baseline = FakeFez device-noise model ``dry_c8_fez``.
    All three deltas are written, each labelled. The strongest baseline is the
    headline; the weak one is reported alongside and labelled weak.
P9. >= 5 replicate arms per reported cell. Replicates here are (hidden shift) x
    (independent device run): 6, 6 and 4 at N = 4, 8, 16. N = 16 has only 4 and is
    flagged ``below_min_replicates`` in the CSV. This is a limit of the stored data
    and cannot be fixed without new QPU time.
P10. NOT RECORDED IS A FINDING. Any requested field absent from the stored
    artifacts is written as the literal string ``NOT_RECORDED`` and itemised in
    ``hardware_provenance_audit.csv``. Nothing is reconstructed or approximated.
    Where a field is only BOUNDED by what was stored (e.g. calibration age is
    bounded by the job submission timestamp), the bound goes in a separate
    ``*_bound`` column so a naive reader cannot mistake it for a measurement.

OUTPUTS (all NEW files, none overwritten)
-----------------------------------------
    results/ibm_campaign/hardware_stats.csv               per-PUB statistics table
    results/ibm_campaign/hardware_information_finiteN.csv per (run, mode, N) info
    results/ibm_campaign/hardware_information_fit.csv     the finite-N fit
    results/ibm_campaign/hardware_provenance_audit.csv    recorded / NOT_RECORDED

Run:
    PYTHONPATH=. .venv314/bin/python -u -m experiments.hardware_ibm.ibm_campaign.hardware_stats
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

from ..readout import readout_from_counts, _split_counts, _bit_of_site

# ---------------------------------------------------------------------------
# pre-registered constants (P3)
# ---------------------------------------------------------------------------
B_BOOT = 10000
BOOT_SEED = 20260722
ALPHA = 0.05
VACUITY_EPS = 1e-10
MIN_REPLICATES = 5

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CAMPAIGN_DIR = os.path.join(REPO, "results", "ibm_campaign")
RUN_TAGS = ["ibm_main", "ibm_c3_probe", "ibm_modef", "ibm_modef2",
            "dry_c8_fez", "dry_c8_ideal"]

NOT_RECORDED = "NOT_RECORDED"


# ---------------------------------------------------------------------------
# counts helpers
# ---------------------------------------------------------------------------
def _strip(counts):
    """(counts_without_reg_order, reg_order, total_shots)."""
    reg = counts.get("__reg_order__")
    c = {k: int(v) for k, v in counts.items() if k != "__reg_order__"}
    return c, reg, int(sum(c.values()))


def _with_reg(c, reg):
    d = dict(c)
    if reg is not None:
        d["__reg_order__"] = reg
    return d


def _resample(c, reg, rng):
    """One multinomial bootstrap resample of a counts dict (P3)."""
    keys = list(c.keys())
    n = int(sum(c.values()))
    p = np.array([c[k] for k in keys], dtype=float) / n
    draw = rng.multinomial(n, p)
    return _with_reg({k: int(v) for k, v in zip(keys, draw) if v > 0}, reg)


def wilson(k, n, alpha=ALPHA):
    """Wilson score interval for a binomial proportion (P2)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    ph = k / n
    d = 1.0 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _reg_value(key, reg_order, name):
    return key.split(" ")[reg_order.index(name)]


def _bit_of_qubit(bits, q):
    return int(bits[::-1][q])


# ---------------------------------------------------------------------------
# per-analysis primary estimators (mirror analyze.py exactly; P1 checks this)
# ---------------------------------------------------------------------------
def _est_string_parity(counts, meta):
    c, reg, total = _strip(counts)
    support = meta["support"]
    num = 0.0
    for key, n in c.items():
        bits = _reg_value(key, reg, "c_str")
        parity = sum(_bit_of_qubit(bits, s) for s in support) % 2
        num += (1.0 if parity == 0 else -1.0) * n
    return num / total if total else float("nan")


def _est_bell_purity(counts, meta):
    c, reg, total = _strip(counts)
    n_pairs = meta["n_pairs"]
    num = 0.0
    for key, n in c.items():
        rev = _reg_value(key, reg, "c_sys")[::-1]
        v = 1.0
        for i in range(n_pairs):
            a = int(rev[2 * i]); b = int(rev[2 * i + 1])
            v *= (1.0 if (a & b) == 0 else -1.0)
        num += v * n
    return num / total if total else float("nan")


def _est_direct_expval(counts, meta):
    c, reg, total = _strip(counts)
    n_data = meta["n_data"]; obs = meta["observable_sites"]
    num = 0.0
    for key, n in c.items():
        data_bits = _reg_value(key, reg, "c_data")
        eig = 1.0
        for s in obs:
            eig *= 1.0 - 2.0 * _bit_of_site(data_bits, s, n_data)
        num += eig * n
    return num / total if total else float("nan")


def _est_postsel(counts, meta):
    r = readout_from_counts(counts, meta["n_data"], meta["n_frame"],
                            observable_sites=meta["observable_sites"])
    return r


def _c8_dist(counts, N):
    """Frame outcome distribution over N bins (mirrors analyze._c8_fourier_row)."""
    c, _, _ = _strip(counts)
    dist = np.zeros(N, dtype=float); total = 0
    for key, n in c.items():
        bits = key.replace(" ", "")
        try:
            k = int(bits, 2)
        except ValueError:
            continue
        if 0 <= k < N:
            dist[k] += n; total += n
    return dist, total


def _shannon_bits(p):
    q = p[p > 0]
    return float(-np.sum(q * np.log2(q)))


def _info_bits(dist, total, N):
    """P5: R = log2 N - H(K|s), plug-in and Miller-Madow corrected."""
    if total <= 0:
        return float("nan"), float("nan")
    p = dist / total
    h = _shannon_bits(p)
    k_obs = int(np.sum(dist > 0))
    h_mm = h + (k_obs - 1) / (2.0 * total * math.log(2.0))
    return math.log2(N) - h, math.log2(N) - h_mm


# ---------------------------------------------------------------------------
# circuit / calibration provenance
# ---------------------------------------------------------------------------
def _load_run(tag):
    d = os.path.join(CAMPAIGN_DIR, tag)
    with open(os.path.join(d, "pub_index.json")) as f:
        pub_index = json.load(f)
    with open(os.path.join(d, "raw_counts.json")) as f:
        counts = json.load(f)
    with open(os.path.join(d, "manifest.json")) as f:
        manifest = json.load(f)
    circuits = None
    try:
        from qiskit import qpy
        with open(os.path.join(d, "circuits_transpiled.qpy"), "rb") as f:
            circuits = qpy.load(f)
    except Exception as e:                                   # pragma: no cover
        print(f"[warn] {tag}: transpiled qpy unreadable ({e}); gate counts -> {NOT_RECORDED}")
    stored = {}
    adir = os.path.join(d, "analysis")
    if os.path.isdir(adir):
        for fn in sorted(os.listdir(adir)):
            if not fn.endswith(".csv"):
                continue
            with open(os.path.join(adir, fn)) as f:
                for row in csv.DictReader(f):
                    if row.get("pub_id"):
                        stored[row["pub_id"]] = row
    return pub_index, counts, manifest, circuits, stored


def _circuit_stats(circ):
    """Total depth, 2q depth, per-gate counts, and the physical 2q edges used."""
    if circ is None:
        return None
    ops = {k: int(v) for k, v in circ.count_ops().items()}
    edges = set()
    n2q = 0
    for instr in circ.data:
        if instr.operation.num_qubits == 2:
            n2q += 1
            qs = tuple(sorted(circ.find_bit(b).index for b in instr.qubits))
            edges.add(qs)
    return {
        "depth_total": int(circ.depth()),
        "depth_2q": int(circ.depth(lambda i: i.operation.num_qubits == 2)),
        "n_2q_gates": n2q,
        "n_cz": ops.get("cz", 0), "n_ecr": ops.get("ecr", 0), "n_cx": ops.get("cx", 0),
        "n_sx": ops.get("sx", 0), "n_rz": ops.get("rz", 0), "n_x": ops.get("x", 0),
        "n_measure": ops.get("measure", 0),
        "edges": sorted(edges),
        "circuit_width_qubits": int(circ.num_qubits),
    }


def _cal_for(manifest, layout, edges):
    """Calibration numbers for exactly the qubits/edges this PUB used."""
    cal = manifest.get("calibration") or {}
    qp = cal.get("qubit_properties") or {}
    tq = cal.get("two_qubit_gates") or {}
    out = {
        "calibration_backend": cal.get("backend", NOT_RECORDED),
        "calibration_source": cal.get("used_qubits_source", cal.get("kind", NOT_RECORDED)),
        "calibration_date": NOT_RECORDED,                       # P10 -- never stored
        "calibration_T1_T2_frequency": NOT_RECORDED,            # P10 -- never stored
    }
    if not qp:
        out.update({"max_measure_error": "", "mean_measure_error": "",
                    "max_sx_error": "", "n_qubits_with_calibration": 0})
    else:
        me = [qp[str(q)]["measure_error"] for q in (layout or []) if str(q) in qp
              and "measure_error" in qp[str(q)]]
        sx = [qp[str(q)]["sx_error"] for q in (layout or []) if str(q) in qp
              and "sx_error" in qp[str(q)]]
        out.update({
            "max_measure_error": max(me) if me else "",
            "mean_measure_error": float(np.mean(me)) if me else "",
            "max_sx_error": max(sx) if sx else "",
            "n_qubits_with_calibration": len(me),
        })
    errs = []
    for (a, b) in (edges or []):
        e = tq.get(f"{a}_{b}") or tq.get(f"{b}_{a}") or {}
        for g in ("cz_error", "ecr_error", "cx_error"):
            if g in e:
                errs.append(e[g]); break
    out.update({
        "n_2q_edges_used": len(edges or []),
        "n_2q_edges_with_calibration": len(errs),
        "max_2q_gate_error": max(errs) if errs else "",
        "mean_2q_gate_error": float(np.mean(errs)) if errs else "",
    })
    return out


# ---------------------------------------------------------------------------
# main per-PUB table
# ---------------------------------------------------------------------------
def build_stats_rows():
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    info_arms = []          # feeds the finite-N information fit
    pub_seen = defaultdict(list)

    loaded = {}
    for tag in RUN_TAGS:
        loaded[tag] = _load_run(tag)
        for p in loaded[tag][0]:
            pub_seen[p["pub_id"]].append(tag)

    for tag in RUN_TAGS:
        pub_index, counts_by_id, manifest, circuits, stored = loaded[tag]
        job = manifest.get("job_id") or NOT_RECORDED
        backend = manifest.get("backend_label", NOT_RECORDED)
        usage = manifest.get("usage") or {}
        metrics = (usage.get("metrics") or {}) if isinstance(usage, dict) else {}
        ts = (metrics.get("timestamps") or {}) if isinstance(metrics, dict) else {}
        qpu_s = ""
        if isinstance(usage, dict):
            u = usage.get("usage")
            qpu_s = u if isinstance(u, (int, float)) else ""
        is_device = str(backend).startswith("IBM:")

        for idx, rec in enumerate(pub_index):
            pid = rec["pub_id"]
            counts = counts_by_id.get(pid)
            if counts is None:
                continue
            c, reg, total = _strip(counts)
            meta = rec.get("meta", {}); theory = rec.get("theory", {})
            analysis = rec["analysis"]
            layout = rec.get("final_index_layout")
            cs = _circuit_stats(circuits[idx]) if circuits is not None else None
            cal = _cal_for(manifest, layout, (cs or {}).get("edges", []))

            row = {
                "run_tag": tag,
                "row_kind": "pub",
                "pub_id": pid,
                "experiment": rec["experiment"],
                "analysis": analysis,
                "backend_label": backend,
                "backend_is_real_device": int(is_device),
                "job_id": job,
                "job_created_utc": ts.get("created", NOT_RECORDED),
                "job_finished_utc": ts.get("finished", NOT_RECORDED),
                "job_billed_qpu_seconds": qpu_s,
                "artifact_timestamp_utc": manifest.get("timestamp_utc", NOT_RECORDED),
                "shots_requested": rec.get("shots", ""),
                "shots_returned": total,
                "n_distinct_outcomes": len(c),
                "qubit_layout_physical": json.dumps(layout) if layout is not None else NOT_RECORDED,
                "n_qubits_used": (len(layout) if layout is not None else ""),
                "register_order": json.dumps(reg) if reg else NOT_RECORDED,
                "raw_counts_json": json.dumps(c, sort_keys=True),
            }
            if cs is not None:
                row.update({k: v for k, v in cs.items() if k != "edges"})
                row["two_qubit_edges_used"] = json.dumps([list(e) for e in cs["edges"]])
            else:
                for k in ("depth_total", "depth_2q", "n_2q_gates", "n_cz", "n_ecr",
                          "n_cx", "n_sx", "n_rz", "n_x", "n_measure",
                          "circuit_width_qubits"):
                    row[k] = NOT_RECORDED
                row["two_qubit_edges_used"] = NOT_RECORDED
            row["depth_2q_stored_in_pub_index"] = rec.get("transpiled_2q_depth", "")
            row.update(cal)
            # --- error mitigation (P10): nothing was stored; code passed no options
            row.update({
                "mitigation_resilience_level": NOT_RECORDED,
                "mitigation_dynamical_decoupling": NOT_RECORDED,
                "mitigation_twirling": NOT_RECORDED,
                "mitigation_measurement_mitigation": NOT_RECORDED,
                "mitigation_options_passed_in_code":
                    "none: SamplerV2(mode=batch) constructed with no options "
                    "(runner.run_ibm); resolved server-side defaults were not stored",
                "analysis_level_postselection": int(analysis == "cqfa_postsel"),
            })
            # --- repeats
            others = [t for t in pub_seen[pid] if t != tag]
            device_repeats = [t for t in pub_seen[pid]
                              if str(loaded[t][2].get("backend_label", "")).startswith("IBM:")]
            row.update({
                "repeat_group_id": pid,
                "n_runs_with_this_pub_id": len(pub_seen[pid]),
                "other_runs_with_this_pub_id": json.dumps(others),
                "n_device_runs_with_this_pub_id": len(device_repeats),
                "is_device_repeat": int(len(device_repeats) > 1 and is_device),
            })

            # ---------------- primary quantity + uncertainty ----------------
            prim_name = prim = ""
            lo = hi = ""
            ci_method = ""
            stored_val = ""
            signal = ""
            extra = {}
            cmp_val = None          # value to compare against stored_val (P1)

            if analysis == "string_parity":
                prim_name = "string_order_hat"
                prim = _est_string_parity(counts, meta)
                boots = [_est_string_parity(_resample(c, reg, rng), meta)
                         for _ in range(B_BOOT)]
                lo, hi = np.percentile(boots, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
                ci_method = "multinomial_bootstrap"
                stored_val = stored.get(pid, {}).get("string_order_hat", "")
                signal = abs(theory.get("string_order", 0.0))
                extra["exact_value"] = theory.get("string_order", "")

            elif analysis == "bell_purity":
                prim_name = "purity_hat"
                prim = _est_bell_purity(counts, meta)
                boots = [_est_bell_purity(_resample(c, reg, rng), meta)
                         for _ in range(B_BOOT)]
                lo, hi = np.percentile(boots, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
                ci_method = "multinomial_bootstrap"
                stored_val = stored.get(pid, {}).get("purity_hat", "")
                signal = abs(theory.get("purity", 0.0))
                extra["exact_value"] = theory.get("purity", "")

            elif analysis == "direct_expval":
                prim_name = "direct_expval_hat"
                prim = _est_direct_expval(counts, meta)
                boots = [_est_direct_expval(_resample(c, reg, rng), meta)
                         for _ in range(B_BOOT)]
                lo, hi = np.percentile(boots, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
                ci_method = "multinomial_bootstrap"
                signal = abs(theory.get("arm_value", 0.0))
                extra["exact_value"] = theory.get("arm_value", "")

            elif analysis == "cqfa_postsel":
                r = _est_postsel(counts, meta)
                prim_name = "f_hat_postsel"
                prim = r["f_hat_postsel"]
                boots = [_est_postsel(_resample(c, reg, rng), meta)["f_hat_postsel"]
                         for _ in range(B_BOOT)]
                boots = np.array(boots, dtype=float)
                boots = boots[~np.isnan(boots)]
                lo, hi = np.percentile(boots, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
                ci_method = "multinomial_bootstrap"
                stored_val = stored.get(pid, {}).get("f_hat_postsel", "")
                fA = meta.get("f_A_theory", theory.get("f_A", 0.0))
                signal = abs(fA if fA is not None else 0.0)
                acc_lo, acc_hi = wilson(int(r["n_accepted"]), int(r["n_total"]))
                extra.update({
                    "exact_value": theory.get("f_A", ""),
                    "f_hat_raw": r["f_hat_raw"],
                    "accept_rate": r["accept_rate"],
                    "accept_rate_ci_lo_wilson": acc_lo,
                    "accept_rate_ci_hi_wilson": acc_hi,
                    "n_accepted": r["n_accepted"],
                })

            elif analysis == "fourier_peak":
                N = int(meta["N"]); s = int(meta["hidden_shift"])
                dist, tot = _c8_dist(counts, N)
                p = dist / tot if tot else dist
                argmax_k = int(np.argmax(p))
                p_arg = float(p[argmax_k]); p_true = float(p[s])
                tv = float(0.5 * np.sum(np.abs(p - 1.0 / N)))
                r_pl, r_mm = _info_bits(dist, tot, N)
                bt_tv, bt_pl, bt_mm = [], [], []
                for _ in range(B_BOOT):
                    d2, t2 = _c8_dist(_resample(c, reg, rng), N)
                    p2 = d2 / t2
                    bt_tv.append(0.5 * np.sum(np.abs(p2 - 1.0 / N)))
                    a, b = _info_bits(d2, t2, N)
                    bt_pl.append(a); bt_mm.append(b)
                q = [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)]
                tv_lo, tv_hi = np.percentile(bt_tv, q)
                pl_lo, pl_hi = np.percentile(bt_pl, q)
                mm_lo, mm_hi = np.percentile(bt_mm, q)
                t_lo, t_hi = wilson(int(round(p_true * tot)), tot)
                a_lo, a_hi = wilson(int(round(p_arg * tot)), tot)
                prim_name = "p_at_true_shift"
                prim = p_true
                lo, hi = t_lo, t_hi
                ci_method = "wilson (pre-registered bin k=s)"
                # stored c8.csv:peak_prob is P(argmax) -- compare like with like (P4)
                stored_val = stored.get(pid, {}).get("peak_prob", "")
                cmp_val = p_arg
                ideal_peak = theory.get("ideal_peak_prob", float("nan"))
                signal = abs(float(ideal_peak) - 1.0 / N)
                extra.update({
                    "exact_value": ideal_peak,
                    "mode": meta["mode"], "N": N, "n_frame": meta["n_frame"],
                    "hidden_shift": s,
                    "p_at_argmax_DATA_SELECTED": p_arg,
                    "p_at_argmax_ci_lo_wilson": a_lo,
                    "p_at_argmax_ci_hi_wilson": a_hi,
                    "argmax_k": argmax_k,
                    "recovered_shift_correct": int(argmax_k == s),
                    "tv_from_uniform": tv,
                    "tv_ci_lo_boot": tv_lo, "tv_ci_hi_boot": tv_hi,
                    "info_bits_plugin": r_pl,
                    "info_bits_plugin_ci_lo": pl_lo, "info_bits_plugin_ci_hi": pl_hi,
                    "info_bits_millermadow": r_mm,
                    "info_bits_mm_ci_lo": mm_lo, "info_bits_mm_ci_hi": mm_hi,
                    "info_bits_max_possible_log2N": math.log2(N),
                })
                info_arms.append({
                    "run_tag": tag, "backend_label": backend, "is_device": is_device,
                    "mode": meta["mode"], "N": N, "n_frame": meta["n_frame"],
                    "hidden_shift": s, "shots": tot,
                    "info_bits_mm": r_mm, "info_bits_plugin": r_pl,
                    "info_mm_boot": np.array(bt_mm, dtype=float),
                    "p_true": p_true, "dist": dist.copy(),
                    "signal_scale": signal,
                })

            elif analysis == "shadow_snapshot":
                label = {int(k): v for k, v in theory["string_label"].items()}
                bases = meta["bases"]; support = [int(x) for x in meta["support"]]
                code = {"X": 0, "Y": 1, "Z": 2}
                hit = int(all(bases[s_] == code[label[s_]] for s_ in support))
                prim_name = "shadow_snapshot_hit"
                prim = hit
                ci_method = "none (single-shot PUB; CI only at the aggregate row)"
                signal = 1.0 / float(meta["three_pow_w"])
                extra["expected_hit_rate"] = 1.0 / float(meta["three_pow_w"])
            else:
                prim_name = analysis; prim = ""

            row["primary_quantity"] = prim_name
            row["primary_value"] = prim
            row["primary_ci_lo"] = lo
            row["primary_ci_hi"] = hi
            row["ci_method"] = ci_method
            row["ci_level"] = 1 - ALPHA
            row["bootstrap_B"] = B_BOOT if "bootstrap" in ci_method else ""
            row["bootstrap_seed"] = BOOT_SEED if "bootstrap" in ci_method else ""
            row["stored_csv_value"] = stored_val
            row["stored_csv_compared_against"] = (
                "p_at_argmax" if cmp_val is not None else prim_name)
            _cmp = prim if cmp_val is None else cmp_val
            try:
                row["matches_stored_csv"] = int(
                    abs(float(stored_val) - float(_cmp)) <= 1e-12) if stored_val != "" else ""
            except (TypeError, ValueError):
                row["matches_stored_csv"] = ""
            row["label_signal_scale"] = signal
            if signal != "" and float(signal) < VACUITY_EPS:
                row["vacuity_flag"] = ("null_control_by_design"
                                       if meta.get("mode") == "B" else "vacuous_excluded")
            else:
                row["vacuity_flag"] = "ok"
            row.update(extra)
            rows.append(row)

    # ---- aggregate row: C1 shadow hit rate with a Wilson CI (P2) ----
    sh = [r for r in rows if r["analysis"] == "shadow_snapshot"]
    if sh:
        k = sum(int(r["primary_value"]) for r in sh)
        n = len(sh)
        lo, hi = wilson(k, n)
        rows.append({
            "run_tag": sh[0]["run_tag"], "row_kind": "aggregate",
            "pub_id": "C1_shadow_n4_AGGREGATE", "experiment": "C1_shadow",
            "analysis": "shadow_snapshot_aggregate",
            "backend_label": sh[0]["backend_label"], "job_id": sh[0]["job_id"],
            "shots_returned": n, "primary_quantity": "shadow_hit_rate",
            "primary_value": k / n, "primary_ci_lo": lo, "primary_ci_hi": hi,
            "ci_method": "wilson", "ci_level": 1 - ALPHA,
            "n_hits": k, "n_snapshots": n,
            "expected_hit_rate": float(sh[0]["expected_hit_rate"]),
            "label_signal_scale": float(sh[0]["expected_hit_rate"]),
            "vacuity_flag": "ok",
            "calibration_date": NOT_RECORDED,
        })
    return rows, info_arms


# ---------------------------------------------------------------------------
# finite-N information table + fit (P5, P6, P8, P9)
# ---------------------------------------------------------------------------
def build_information_tables(info_arms):
    q = [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)]
    groups = defaultdict(list)
    for a in info_arms:
        groups[(a["run_tag"], a["mode"], a["N"])].append(a)

    fin_rows = []
    for (tag, mode, N), arms in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        vals = np.array([a["info_bits_mm"] for a in arms], dtype=float)
        n = len(vals)
        mean = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        # joint bootstrap of the group mean (shot noise), B replicates
        joint = np.mean(np.vstack([a["info_mm_boot"] for a in arms]), axis=0)
        b_lo, b_hi = np.percentile(joint, q)
        # pooled mutual information I(S;K), capped at log2(n_shifts) -- P5
        stack = np.vstack([a["dist"] for a in arms])
        pk = stack.sum(axis=0); pk = pk / pk.sum()
        h_marg = _shannon_bits(pk)
        h_cond = float(np.mean([_shannon_bits(a["dist"] / a["dist"].sum()) for a in arms]))
        fin_rows.append({
            "run_tag": tag, "backend_label": arms[0]["backend_label"],
            "is_real_device": int(arms[0]["is_device"]), "mode": mode,
            "N": N, "n_frame": arms[0]["n_frame"], "log2N": math.log2(N),
            "n_arms": n, "shifts_measured": json.dumps([a["hidden_shift"] for a in arms]),
            "shots_per_arm": arms[0]["shots"],
            "info_bits_mm_mean": mean,
            "info_bits_mm_se_across_arms": se,
            "info_bits_mm_ci_lo_boot": b_lo, "info_bits_mm_ci_hi_boot": b_hi,
            "info_bits_plugin_mean": float(np.mean([a["info_bits_plugin"] for a in arms])),
            "info_bits_max_possible_log2N": math.log2(N),
            "mutual_info_S_K_bits": h_marg - h_cond,
            "mutual_info_cap_log2_n_shifts": math.log2(n),
            "mutual_info_capped_by_design": 1,
            "mean_p_at_true_shift": float(np.mean([a["p_true"] for a in arms])),
            "label_signal_scale": arms[0]["signal_scale"],
            "vacuity_flag": ("null_control_by_design" if mode == "B" else "ok"),
            "below_min_replicates": int(n < MIN_REPLICATES),
            "min_replicates_rule": MIN_REPLICATES,
        })

    # ---- pooled device arms (the two independent ibm_fez runs) + baselines ----
    dev = defaultdict(list)
    base = {}
    for a in info_arms:
        if a["mode"] != "F":
            continue
        if a["is_device"]:
            dev[a["N"]].append(a)
        else:
            base.setdefault((a["run_tag"], a["N"]), []).append(a)

    fit_rows = []
    Ns = sorted(dev)
    x = np.array([math.log2(N) for N in Ns], dtype=float)
    y, sey, narm = [], [], []
    boot_mat = []
    for N in Ns:
        arms = dev[N]
        v = np.array([a["info_bits_mm"] for a in arms], dtype=float)
        y.append(float(np.mean(v)))
        sey.append(float(np.std(v, ddof=1) / math.sqrt(len(v))))
        narm.append(len(v))
        boot_mat.append(np.mean(np.vstack([a["info_mm_boot"] for a in arms]), axis=0))
    y = np.array(y); sey = np.array(sey); boot_mat = np.vstack(boot_mat)

    def wls(xx, yy, ww):
        W = np.diag(ww)
        A = np.vstack([xx, np.ones_like(xx)]).T
        beta = np.linalg.solve(A.T @ W @ A, A.T @ W @ yy)
        return float(beta[0]), float(beta[1])

    w = 1.0 / np.maximum(sey, 1e-12) ** 2
    slope, intercept = wls(x, y, w)
    bs, bi = [], []
    for b in range(B_BOOT):
        s_, i_ = wls(x, boot_mat[:, b], w)
        bs.append(s_); bi.append(i_)
    s_lo, s_hi = np.percentile(bs, q)
    i_lo, i_hi = np.percentile(bi, q)
    resid = y - (slope * x + intercept)

    # baselines (P8)
    def base_mean(tag, N):
        arms = base.get((tag, N), [])
        return float(np.mean([a["info_bits_mm"] for a in arms])) if arms else float("nan")

    for j, N in enumerate(Ns):
        ideal = base_mean("dry_c8_ideal", N)
        fake = base_mean("dry_c8_fez", N)
        fit_rows.append({
            "quantity": "info_bits_mm (R = log2 N - H_MM(K|s))",
            "point": f"N={N}", "N": N, "log2N": math.log2(N),
            "n_device_arms": narm[j],
            "below_min_replicates": int(narm[j] < MIN_REPLICATES),
            "device_mean": y[j], "device_se_across_arms": sey[j],
            "device_ci_lo_boot": float(np.percentile(boot_mat[j], q[0])),
            "device_ci_hi_boot": float(np.percentile(boot_mat[j], q[1])),
            "fit_predicted": float(slope * x[j] + intercept),
            "residual": float(resid[j]),
            "baseline_weak_blind_chance": 0.0,
            "delta_vs_weak_baseline_blind": y[j] - 0.0,
            "baseline_strong_ideal_sim": ideal,
            "delta_vs_strong_baseline_ideal": y[j] - ideal,
            "baseline_strongest_fakefez_noisemodel": fake,
            "delta_vs_strongest_baseline_fakefez": y[j] - fake,
            "max_possible_log2N": math.log2(N),
            "fraction_of_max_recovered": y[j] / math.log2(N),
            "slope": "", "intercept": "",
        })
    fit_rows.append({
        "quantity": "WLS fit  R = slope*log2(N) + intercept",
        "point": "FIT", "N": "", "log2N": "",
        "n_device_arms": int(sum(narm)),
        "below_min_replicates": int(any(nn < MIN_REPLICATES for nn in narm)),
        "device_mean": "", "device_se_across_arms": "",
        "device_ci_lo_boot": "", "device_ci_hi_boot": "",
        "fit_predicted": "", "residual": float(np.sqrt(np.mean(resid ** 2))),
        "baseline_weak_blind_chance": 0.0,
        "delta_vs_weak_baseline_blind": "",
        "baseline_strong_ideal_sim": "", "delta_vs_strong_baseline_ideal": "",
        "baseline_strongest_fakefez_noisemodel": "",
        "delta_vs_strongest_baseline_fakefez": "",
        "max_possible_log2N": "", "fraction_of_max_recovered": "",
        "slope": slope, "intercept": intercept,
        "slope_ci_lo": s_lo, "slope_ci_hi": s_hi,
        "intercept_ci_lo": i_lo, "intercept_ci_hi": i_hi,
        "ideal_slope_would_be": 1.0, "ideal_intercept_would_be": 0.0,
        "N_values_covered": json.dumps(Ns),
        "n_frame_qubits_covered": json.dumps([int(math.log2(N)) for N in Ns]),
        "n_distinct_N": len(Ns),
        "fit_scope_statement":
            "DESCRIPTIVE, LOCAL FIT over N in {4,8,16} (2-4 frame qubits, 3 points) "
            "from two ibm_fez runs. Three points cannot discriminate log2 N from any "
            "other function agreeing on them. NO asymptotic claim is made or supported.",
        "rmse_bits": float(np.sqrt(np.mean(resid ** 2))),
        "bootstrap_B": B_BOOT, "bootstrap_seed": BOOT_SEED, "ci_level": 1 - ALPHA,
    })
    return fin_rows, fit_rows


# ---------------------------------------------------------------------------
# provenance audit (P10)
# ---------------------------------------------------------------------------
AUDIT = [
    ("raw outcome histograms", "RECORDED", "raw_counts.json per run dir -> hardware_stats.csv:raw_counts_json",
     "Full per-PUB counts dicts for all 582 PUBs across 6 runs."),
    ("shots per PUB", "RECORDED", "pub_index.json:shots; recomputed as shots_returned",
     "Requested and returned shots both written; they agree everywhere."),
    ("confidence intervals", "COMPUTED_HERE", "hardware_stats.csv:primary_ci_lo/hi, ci_method",
     "Wilson for binomial proportions; multinomial bootstrap (B=10000, seed=20260722) otherwise."),
    ("qubit layout (physical)", "RECORDED (device runs only)", "pub_index.json:final_index_layout",
     "Per-PUB physical layout stored for the 4 IBM runs; NULL for the two offline runs "
     "(no physical layout exists offline)."),
    ("circuit depth (total)", "RECOVERABLE", "circuits_transpiled.qpy -> depth_total",
     "Not stored as a field, but recomputed exactly from the saved transpiled circuits."),
    ("two-qubit depth", "RECORDED", "pub_index.json:transpiled_2q_depth",
     "Stored per PUB; independently recomputed here and verified to match on all 582 PUBs."),
    ("two-qubit gate COUNT", "RECOVERABLE", "circuits_transpiled.qpy -> n_2q_gates, n_cz",
     "Not stored as a field; recomputed exactly from the saved transpiled circuits."),
    ("two-qubit edges used", "RECOVERABLE", "circuits_transpiled.qpy -> two_qubit_edges_used",
     "Physical edge list recomputed per PUB."),
    ("gate error rates (sx, x, measure)", "RECORDED (device runs only)",
     "manifest.json:calibration.qubit_properties",
     "Per used qubit. Offline runs store calibration.kind='offline' with no numbers."),
    ("two-qubit gate error rates", "RECORDED (device runs only)",
     "manifest.json:calibration.two_qubit_gates", "cz_error and cz_duration per used edge."),
    ("CALIBRATION DATE / timestamp", NOT_RECORDED, "-- absent from every manifest --",
     "The calibration snapshot carries NO timestamp. It was taken in-process at job "
     "submission (runner._calibration_snapshot is called on the live backend target "
     "before Batch submission), so its age is BOUNDED by job_created_utc, but the "
     "actual last-calibration time of the device is NOT recoverable. Reported as a "
     "bound in job_created_utc, never as a measurement."),
    ("T1 / T2 / qubit frequency", NOT_RECORDED, "-- absent from every manifest --",
     "runner._calibration_snapshot attempts T1/T2/frequency inside a try/except; the "
     "attempt evidently failed on this backend/target version and NO manifest contains "
     "these fields. Not reconstructable without new device queries."),
    ("error-mitigation settings", NOT_RECORDED, "-- absent from every manifest --",
     "No resilience_level / dynamical_decoupling / twirling / measurement-mitigation "
     "values were stored. Source (runner.run_ibm) shows SamplerV2(mode=batch) built "
     "with NO options, so server-side defaults applied, but the RESOLVED values were "
     "never captured and are not reconstructed here."),
    ("post-selection (analysis-level mitigation)", "RECORDED",
     "hardware_stats.csv:analysis_level_postselection, accept_rate + Wilson CI",
     "C3/C5/C6/C7 post-select on the frame register; acceptance rate is measured."),
    ("repeated jobs", "RECORDED", "hardware_stats.csv:repeat_group_id, n_device_runs_with_this_pub_id",
     "16 C8 PUB ids appear in 4 runs each (2 IBM device runs ibm_modef/ibm_modef2 on "
     "DIFFERENT layouts, plus ideal and FakeFez offline runs). C1-C7 have NO device "
     "repeat: each was run exactly once."),
    ("job ids / billed QPU seconds", "RECORDED", "manifest.json:job_id, usage.metrics",
     "4 device jobs; billed 9 + 48 + 19 + 19 = 95 quantum-seconds."),
    ("readout-error mitigation matrix", NOT_RECORDED, "-- never collected --",
     "No calibration-matrix job was run, so no measurement-error unfolding is possible "
     "from stored data."),
    ("randomized benchmarking / device-level fidelity", NOT_RECORDED, "-- never collected --",
     "Only the vendor-reported per-gate errors above; no independent RB was run."),
    ("per-shot / per-job timestamps within a job", NOT_RECORDED, "-- absent --",
     "Only job-level created/running/finished timestamps exist; PUBs inside a batched "
     "job carry no individual timing, so drift within a job cannot be assessed."),
]


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------
def _write_csv(path, rows):
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite existing results file: {path}")
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path}  ({len(rows)} rows x {len(cols)} cols)")
    return path


def build_reproducibility_rows():
    """PAIRED run-to-run reproducibility of the Mode-F information quantity.

    Reads back the per-PUB hardware_stats.csv (provenance: every number here is a
    cell of that file) and forms PAIRED per-arm deltas between the two independent
    ibm_fez device runs, which sat on DIFFERENT physical layouts. Difference first,
    then average (P: paired deltas, never a difference of group means).
    se = std(ddof=1)/sqrt(n).

    This is the quantity that determines whether a single run's shot-noise error
    bar is a complete uncertainty on the reported value.
    """
    path = os.path.join(CAMPAIGN_DIR, "hardware_stats.csv")
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["analysis"] == "fourier_peak"]
    by = {}
    for r in rows:
        by[(r["run_tag"], r["mode"], int(r["N"]), int(r["hidden_shift"]))] = r
    out = []
    arms = sorted({(m, N, s) for (t, m, N, s) in by if t == "ibm_modef"})
    for (m, N, s) in arms:
        a = by.get(("ibm_modef", m, N, s)); b = by.get(("ibm_modef2", m, N, s))
        if a is None or b is None:
            continue
        ia, ib = float(a["info_bits_millermadow"]), float(b["info_bits_millermadow"])
        # primary_quantity is 'p_at_true_shift' for every fourier_peak row (P4)
        assert a["primary_quantity"] == "p_at_true_shift" == b["primary_quantity"]
        pa, pb = float(a["primary_value"]), float(b["primary_value"])
        # shot-noise half-width of run 1's own CI, for the comparison below
        hw = (float(a["info_bits_mm_ci_hi"]) - float(a["info_bits_mm_ci_lo"])) / 2.0
        out.append({
            "row_kind": "paired_arm", "mode": m, "N": N, "hidden_shift": s,
            "run1_tag": "ibm_modef", "run2_tag": "ibm_modef2",
            "run1_layout": a["qubit_layout_physical"], "run2_layout": b["qubit_layout_physical"],
            "same_layout": int(a["qubit_layout_physical"] == b["qubit_layout_physical"]),
            "run1_info_bits_mm": ia, "run2_info_bits_mm": ib,
            "paired_delta_info_bits": ib - ia,
            "abs_paired_delta_info_bits": abs(ib - ia),
            "run1_shotnoise_ci_halfwidth_bits": hw,
            "delta_exceeds_run1_shotnoise_ci": int(abs(ib - ia) > hw),
            "run1_p_at_true_shift": pa, "run2_p_at_true_shift": pb,
            "paired_delta_p_at_true_shift": pb - pa,
            "run1_depth_2q": a["depth_2q"], "run2_depth_2q": b["depth_2q"],
            "run1_n_2q_gates": a["n_2q_gates"], "run2_n_2q_gates": b["n_2q_gates"],
            "run1_max_2q_gate_error": a["max_2q_gate_error"],
            "run2_max_2q_gate_error": b["max_2q_gate_error"],
        })
    for mode in ("F", "B"):
        d = [r["paired_delta_info_bits"] for r in out if r["mode"] == mode]
        ad = [r["abs_paired_delta_info_bits"] for r in out if r["mode"] == mode]
        hws = [r["run1_shotnoise_ci_halfwidth_bits"] for r in out if r["mode"] == mode]
        n = len(d)
        if n == 0:
            continue
        out.append({
            "row_kind": "summary", "mode": mode, "N": "ALL", "hidden_shift": "ALL",
            "run1_tag": "ibm_modef", "run2_tag": "ibm_modef2",
            "n_paired_arms": n,
            "mean_paired_delta_info_bits": float(np.mean(d)),
            "se_paired_delta_info_bits": float(np.std(d, ddof=1) / math.sqrt(n)) if n > 1 else "",
            "mean_abs_paired_delta_info_bits": float(np.mean(ad)),
            "max_abs_paired_delta_info_bits": float(np.max(ad)),
            "mean_shotnoise_ci_halfwidth_bits": float(np.mean(hws)),
            "ratio_mean_abs_delta_to_shotnoise_halfwidth":
                float(np.mean(ad)) / float(np.mean(hws)) if np.mean(hws) > 0 else "",
            "n_arms_exceeding_shotnoise_ci":
                int(sum(r["delta_exceeds_run1_shotnoise_ci"] for r in out
                        if r["row_kind"] == "paired_arm" and r["mode"] == mode)),
            "below_min_replicates": int(n < MIN_REPLICATES),
            "interpretation":
                "run-to-run (layout/calibration) variation vs within-run shot noise; "
                "ratio > 1 means a single run's shot-noise bar UNDERSTATES the true "
                "uncertainty on the reported value",
        })
    return out


def main():
    rows, info_arms = build_stats_rows()
    fin_rows, fit_rows = build_information_tables(info_arms)
    audit_rows = [{"requested_field": a, "status": b, "where": c, "note": d}
                  for (a, b, c, d) in AUDIT]

    paths = []
    paths.append(_write_csv(os.path.join(CAMPAIGN_DIR, "hardware_stats.csv"), rows))
    paths.append(_write_csv(os.path.join(CAMPAIGN_DIR, "hardware_information_finiteN.csv"), fin_rows))
    paths.append(_write_csv(os.path.join(CAMPAIGN_DIR, "hardware_information_fit.csv"), fit_rows))
    paths.append(_write_csv(os.path.join(CAMPAIGN_DIR, "hardware_provenance_audit.csv"), audit_rows))
    paths.append(_write_csv(os.path.join(CAMPAIGN_DIR, "hardware_run_reproducibility.csv"),
                            build_reproducibility_rows()))

    # ---- console self-checks (P1) ----
    checked = [r for r in rows if r.get("matches_stored_csv") not in ("", None)]
    bad = [r for r in checked if int(r["matches_stored_csv"]) == 0]
    print(f"\n[P1] recomputed-vs-stored: {len(checked)} rows checked, {len(bad)} mismatches")
    for r in bad[:10]:
        print(f"     MISMATCH {r['run_tag']}/{r['pub_id']}: "
              f"recomputed={r['primary_value']} stored={r['stored_csv_value']}")
    dep = [r for r in rows if r["row_kind"] == "pub"
           and r.get("depth_2q") not in ("", NOT_RECORDED)]
    dbad = [r for r in dep if int(r["depth_2q"]) != int(r["depth_2q_stored_in_pub_index"])]
    print(f"[P1] recomputed 2q-depth vs stored: {len(dep)} checked, {len(dbad)} mismatches")
    vac = [r for r in rows if r.get("vacuity_flag") == "vacuous_excluded"]
    nul = [r for r in rows if r.get("vacuity_flag") == "null_control_by_design"]
    print(f"[P7] vacuous_excluded={len(vac)}  null_control_by_design={len(nul)}")
    fit = fit_rows[-1]
    print(f"[P6] fit slope={fit['slope']:.4f} "
          f"[{fit['slope_ci_lo']:.4f}, {fit['slope_ci_hi']:.4f}] "
          f"intercept={fit['intercept']:.4f} "
          f"[{fit['intercept_ci_lo']:.4f}, {fit['intercept_ci_hi']:.4f}]  "
          f"over N={fit['N_values_covered']}")
    return paths


if __name__ == "__main__":
    import sys
    if "--repro-only" in sys.argv:
        # writes ONLY the new reproducibility table (reads back hardware_stats.csv)
        _write_csv(os.path.join(CAMPAIGN_DIR, "hardware_run_reproducibility.csv"),
                   build_reproducibility_rows())
    else:
        main()
