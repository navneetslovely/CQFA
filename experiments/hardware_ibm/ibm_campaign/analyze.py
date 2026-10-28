"""
experiments.hardware_ibm.ibm_campaign.analyze
==========================================
Turn raw per-PUB counts into per-experiment CSV rows. PURE and OFFLINE: it reads
only the saved raw_counts + the stored theory, so it re-runs identically on the
device counts long after the job, and the dry run and the real run share one code
path. Reuses the verified ``readout`` conventions for the CQFA circuits.

Dispatch by ``PubSpec.analysis``:
    string_parity   C1 direct <O_string>            -> abs err vs +1 / 0
    shadow_snapshot C1 single-copy shadow snapshots  -> hit rate vs 1/3^w
    bell_purity     C2 Tr(rho^2) via destructive Bell
    cqfa_postsel    C3/C5/C6 f_A (post-sel) vs f_raw, accept rate
    direct_expval   C4 Mode-B orbit arm <U_g psi|M|U_g psi>
(C4 surplus and C5 equivariance are assembled across their grouped PUBs.)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..readout import readout_from_counts, _split_counts, _bit_of_site


_BASIS_CODE = {"X": 0, "Y": 1, "Z": 2}


# ---------------------------------------------------------------------------
# low-level counts helpers
# ---------------------------------------------------------------------------
def _reg_value(key: str, reg_order, name: str) -> str:
    return key.split(" ")[reg_order.index(name)]


def _bit_of_qubit(reg_bits: str, q: int) -> int:
    """Bit of qubit index q in an MSB-first register bitstring (qubit=site here)."""
    return int(reg_bits[::-1][q])


# ---------------------------------------------------------------------------
# C1 direct string readout
# ---------------------------------------------------------------------------
def _c1_string_parity(counts, spec):
    joint, reg_order, total = _split_counts(counts)
    support = spec.meta["support"]
    num = 0.0
    for key, n in joint.items():
        bits = _reg_value(key, reg_order, "c_str")
        parity = sum(_bit_of_qubit(bits, s) for s in support) % 2
        num += (1.0 if parity == 0 else -1.0) * n
    est = num / total if total else float("nan")
    s_exact = spec.theory["string_order"]
    return {
        "pub_id": spec.pub_id, "experiment": "C1", "n": spec.meta["n"],
        "phase": spec.meta["phase"], "weight": spec.theory["weight"],
        "string_order_exact": s_exact, "string_order_hat": est,
        "abs_err": abs(est - s_exact), "shots": total,
    }


def _c1_shadow_aggregate(specs, counts_by_id):
    """Aggregate all C1 shadow snapshots (per n): hit rate vs 1/3^w, variance ~3^w."""
    by_n = defaultdict(list)
    for s in specs:
        if s.analysis != "shadow_snapshot":
            continue
        by_n[s.meta["n"]].append(s)
    rows = []
    for n, group in by_n.items():
        w = group[0].meta["weight"]
        support = [int(s) for s in group[0].meta["support"]]
        label = {int(k): v for k, v in group[0].theory["string_label"].items()}
        vals, hits = [], 0
        for s in group:
            counts = counts_by_id[s.pub_id]
            joint, reg_order, total = _split_counts(counts)
            key = next(iter(joint))                       # single-shot pub
            bits = _reg_value(key, reg_order, "c_str")
            bases = s.meta["bases"]
            v, hit = 1.0, True
            for site in support:
                if bases[site] != _BASIS_CODE[label[site]]:
                    v, hit = 0.0, False
                    break
                v *= 3.0 * (1.0 if _bit_of_qubit(bits, site) == 0 else -1.0)
            vals.append(v)
            hits += int(hit)
        vals = np.array(vals, dtype=float)
        rows.append({
            "experiment": "C1_shadow", "n": n, "weight": w,
            "three_pow_w": 3 ** w, "n_snapshots": len(vals), "n_hits": hits,
            "hit_rate": hits / len(vals) if len(vals) else float("nan"),
            "expected_hit_rate": 1.0 / 3 ** w,
            # budget-limited illustration of the ~1/3^w single-copy hit rate: with a
            # few hundred snapshots these are estimable-but-crude, NOT a measurement of
            # V_shadow=3^w. The quantitative variance law is the simulated many-snapshot
            # simulation (results/spt_measurement_cost/exponential_separation.csv), not this arm.
            "shadow_mean_illustrative": float(np.mean(vals)) if len(vals) else float("nan"),
            "shadow_var_illustrative": float(np.var(vals)) if len(vals) else float("nan"),
            "note": "budget-limited: hit_rate illustrates 1/3^w; variance law is from simulation",
        })
    return rows


# ---------------------------------------------------------------------------
# C2 two-copy purity
# ---------------------------------------------------------------------------
def _c2_bell_purity(counts, spec):
    joint, reg_order, total = _split_counts(counts)
    n_pairs = spec.meta["n_pairs"]
    num = 0.0
    for key, n in joint.items():
        sysbits = _reg_value(key, reg_order, "c_sys")
        rev = sysbits[::-1]                               # rev[j] = classical bit j
        v = 1.0
        for i in range(n_pairs):
            a = int(rev[2 * i]); b = int(rev[2 * i + 1])
            v *= (1.0 if (a & b) == 0 else -1.0)
        num += v * n
    est = num / total if total else float("nan")
    pur = spec.theory["purity"]
    return {
        "pub_id": spec.pub_id, "experiment": "C2", "kind": spec.meta["kind"],
        "theta": spec.meta["theta"], "purity_exact": pur, "purity_hat": est,
        "abs_err": abs(est - pur), "shots": total,
    }


# ---------------------------------------------------------------------------
# C3 / C5 / C6 post-selected CQFA readout
# ---------------------------------------------------------------------------
def _cqfa_postsel_row(counts, spec):
    n_data = spec.meta["n_data"]; n_frame = spec.meta["n_frame"]
    obs_sites = spec.meta["observable_sites"]
    r = readout_from_counts(counts, n_data, n_frame, observable_sites=obs_sites)
    f_exact = spec.theory["f_A"]
    fp, fr = r["f_hat_postsel"], r["f_hat_raw"]
    return {
        "pub_id": spec.pub_id, "experiment": spec.experiment,
        "input_id": spec.meta.get("input_id"),
        "observable": spec.meta.get("observable"),
        "f_A_exact": f_exact, "f_hat_postsel": fp, "f_hat_raw": fr,
        "abs_err_postsel": abs(fp - f_exact) if not np.isnan(fp) else float("nan"),
        "abs_err_raw": abs(fr - f_exact) if not np.isnan(fr) else float("nan"),
        "accept_rate": r["accept_rate"], "n_accepted": r["n_accepted"],
        "p_exact": spec.theory.get("p"), "shots": r["n_total"],
        **{k: spec.meta[k] for k in ("shot_level", "n_layers", "g") if k in spec.meta},
    }


# ---------------------------------------------------------------------------
# C4 Mode-B direct arm + surplus assembly
# ---------------------------------------------------------------------------
def _direct_expval(counts, spec):
    joint, reg_order, total = _split_counts(counts)
    n_data = spec.meta["n_data"]; obs_sites = spec.meta["observable_sites"]
    num = 0.0
    for key, n in joint.items():
        data_bits = _reg_value(key, reg_order, "c_data")
        eig = 1.0
        for s in obs_sites:
            eig *= 1.0 - 2.0 * _bit_of_site(data_bits, s, n_data)
        num += eig * n
    return num / total if total else float("nan")


def _c4_surplus_rows(specs, counts_by_id):
    """Combine Mode A (post-sel) and averaged Mode B arms into surplus per input."""
    groups = defaultdict(lambda: {"A": None, "B": []})
    theory = {}
    for s in specs:
        if s.experiment != "C4":
            continue
        g = s.meta["surplus_group"]
        if s.meta["mode"] == "A":
            r = readout_from_counts(counts_by_id[s.pub_id], s.meta["n_data"],
                                    s.meta["n_frame"],
                                    observable_sites=s.meta["observable_sites"])
            groups[g]["A"] = r["f_hat_postsel"]
            theory[g] = (s.meta["f_A_theory"], s.meta["f_B_theory"],
                         s.meta["surplus_theory"])
        else:
            groups[g]["B"].append(_direct_expval(counts_by_id[s.pub_id], s))
    rows = []
    for g, d in groups.items():
        f_A = d["A"]; f_B = float(np.mean(d["B"])) if d["B"] else float("nan")
        surplus = f_A - f_B
        fAt, fBt, surt = theory[g]
        rows.append({
            "experiment": "C4", "input_id": g,
            "f_A_exact": fAt, "f_B_exact": fBt, "surplus_exact": surt,
            "f_A_hat": f_A, "f_B_hat": f_B, "surplus_hat": surplus,
            "abs_err_surplus": abs(surplus - surt) if surt is not None else float("nan"),
        })
    return rows


def _c5_equivariance_rows(cqfa_rows):
    """Assemble C5: within each input, |f_A(psi) - f_A(U_g psi)| should be ~0."""
    by_input = defaultdict(dict)
    for r in cqfa_rows:
        if r["experiment"] != "C5":
            continue
        by_input[r["input_id"]][r["g"]] = r
    rows = []
    for input_id, arms in by_input.items():
        if 0 not in arms:
            continue
        base = arms[0]["f_hat_postsel"]
        for g, r in sorted(arms.items()):
            rows.append({
                "experiment": "C5", "input_id": input_id, "g": g,
                "f_A_exact": r["f_A_exact"], "f_hat_postsel": r["f_hat_postsel"],
                "equivariance_gap_vs_g0": abs(r["f_hat_postsel"] - base),
                "abs_err_vs_exact": r["abs_err_postsel"],
                "accept_rate": r["accept_rate"], "shots": r["shots"],
            })
    return rows


# ---------------------------------------------------------------------------
# C8 Mode-F frame Fourier readout (hidden-shift recovery vs blind Mode B)
# ---------------------------------------------------------------------------
def _c8_fourier_row(counts, spec):
    """Frame-register outcome distribution -> peak, recovered shift, blindness (TV
    from uniform). Mode F: peak should sit at k=s (recovers the hidden shift);
    Mode B: distribution should be uniform (blind, TV~0)."""
    f = spec.meta["n_frame"]; N = spec.meta["N"]
    s = spec.meta["hidden_shift"]; mode = spec.meta["mode"]
    dist = np.zeros(N, dtype=float); total = 0
    for key, n in counts.items():
        if key == "__reg_order__":
            continue
        bits = key.replace(" ", "")              # single c_frame register (MSB-first)
        try:
            k = int(bits, 2)
        except ValueError:
            continue
        if 0 <= k < N:
            dist[k] += n; total += n
    p = dist / total if total else dist
    peak_k = int(np.argmax(p)); peak_prob = float(p[peak_k])
    tv = float(0.5 * np.sum(np.abs(p - 1.0 / N)))
    row = {
        "pub_id": spec.pub_id, "experiment": "C8", "mode": mode, "N": N,
        "n_frame": f, "hidden_shift": s, "peak_k": peak_k, "peak_prob": peak_prob,
        "recovered_shift": peak_k, "success": int(peak_k == s),
        "tv_from_uniform": tv, "shots": total,
        "ideal_peak_prob": spec.theory.get("ideal_peak_prob"),
        # gate/paper metric: Mode F peak should reach ~1 at k=s; Mode B TV should ~0.
        "abs_err": abs(peak_prob - 1.0) if mode == "F" else tv,
    }
    return row


# ---------------------------------------------------------------------------
# top-level dispatch
# ---------------------------------------------------------------------------
def analyze_all(specs, counts_by_id):
    """Return {experiment_tag: [rows]} for every experiment."""
    out = defaultdict(list)
    cqfa_rows = []
    for s in specs:
        counts = counts_by_id.get(s.pub_id)
        if counts is None:
            continue
        if s.analysis == "string_parity":
            out["C1"].append(_c1_string_parity(counts, s))
        elif s.analysis == "bell_purity":
            out["C2"].append(_c2_bell_purity(counts, s))
        elif s.analysis == "cqfa_postsel":
            row = _cqfa_postsel_row(counts, s)
            cqfa_rows.append(row)
            out[s.experiment].append(row)
        elif s.analysis == "fourier_peak":
            out["C8"].append(_c8_fourier_row(counts, s))
        # shadow_snapshot and direct_expval are aggregated below
    out["C1_shadow"] = _c1_shadow_aggregate(specs, counts_by_id)
    out["C4"] = _c4_surplus_rows(specs, counts_by_id)
    out["C5"] = _c5_equivariance_rows(cqfa_rows)
    return dict(out)
