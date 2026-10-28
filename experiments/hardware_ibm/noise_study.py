
from __future__ import annotations

import os

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from cqfa import readout as Mo
from .cqfa_circuit import (
    build_cqfa_circuit, hardware_representation, numpy_to_qiskit_amplitudes,
    pauli_observable,
)
from .backends import get_backend
from .readout import readout_from_counts, statevector_readout
from ._util import write_csv, bootstrap_ci



GROUP_OBSERVABLE = {
    "z2_rc": {0: "X", 1: "Z"},          # A = X(x)Z ; XZ commutes with A, f_A up to 1
    "z2_pauli": {0: "X"},               # X_0 commutes with X^(x)n
    "z2xz2": {0: "X"},                  # X_0 commutes with the X-strings
}


NOISE_STUDY_FIELDS = [
    "group", "rep_name", "n_data", "n_frame", "backend_label", "noise_model",
    "noise_strength", "input_id", "h", "observable", "theta_id", "shots", "seed",
    "transpiled_2q_depth", "f_A_exact", "f_hat_postsel", "f_hat_raw",
    "abs_err_postsel", "abs_err_raw", "accept_rate", "p_exact", "n_accepted",
]

SUMMARY_FIELDS = [
    "group", "backend_label", "noise_strength", "shots", "n_runs", "n_seeds",
    "n_inputs", "mean_abs_err_postsel", "std_abs_err_postsel", "ci_lo_postsel",
    "ci_hi_postsel", "mean_abs_err_raw", "std_abs_err_raw", "ci_lo_raw",
    "ci_hi_raw", "mean_accept_rate", "mean_n_accepted",
]


# ---------------------------------------------------------------------------
# input generation
# ---------------------------------------------------------------------------
def _inputs_for_group(group, rng, include_random=True):
    """Yield (input_id, h, psi) data states for a group.

    z2xz2 -> SPT ground states for a sweep of transverse field h (+ random).
    z2_rc -> random states (+ a fixed reproducible one).
    """
    inputs = []
    if group == "z2xz2":
        from experiments.spt_measurement_cost import spt_physics_reference as spt
        for h in (0.0, 0.5, 1.0, 2.0):
            psi = np.asarray(spt.ground_state(4, h), dtype=complex)
            psi = psi / np.linalg.norm(psi)
            inputs.append((f"spt_h{h}", h, psi))
    else:
        dim = 4  # z2_rc
        psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        psi /= np.linalg.norm(psi)
        inputs.append(("fixed0", float("nan"), psi))
    if include_random:
        rep = hardware_representation(group, {"z2xz2": 4, "z2_rc": 2}.get(group, 2))
        for k in range(1):
            psi = rng.standard_normal(rep.dim) + 1j * rng.standard_normal(rep.dim)
            psi /= np.linalg.norm(psi)
            inputs.append((f"rand{k}", float("nan"), psi))
    return inputs


# ---------------------------------------------------------------------------
# circuit assembly with state prep
# ---------------------------------------------------------------------------
def _prepared_circuits(rep, psi, theta, n_layers, observable):
   
    n_data = int(round(np.log2(rep.dim)))
    n_frame = rep.n_frame_qubits
    psi_q = numpy_to_qiskit_amplitudes(psi, n_data)
    d = QuantumRegister(n_data, "data")
    fr = QuantumRegister(n_frame, "frame")
    initc = QuantumCircuit(d, fr)
    initc.initialize(psi_q, [d[i] for i in range(n_data)])
    qc_m, _ = build_cqfa_circuit(rep, theta=theta, mode="postselect",
                                 measure=True, n_layers=n_layers,
                                 observable=observable)
    qc_s, _ = build_cqfa_circuit(rep, theta=theta, mode="postselect",
                                 measure=False, n_layers=n_layers)
    return initc.compose(qc_m), initc.compose(qc_s), n_data, n_frame


# ---------------------------------------------------------------------------
# main sweep
# ---------------------------------------------------------------------------
def run_noise_study(groups=("z2_rc", "z2xz2"), depol_ps=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
                    shots_list=(256, 1024, 4096, 16384), n_seeds=5, n_layers=0,
                    use_fake=True, use_theta=False, out_dir=None, verbose=True,
                    max_qubits_fake=None):
    
    fake_for = {"z2_rc": "manila", "z2xz2": "sherbrooke"}
    rows = []
    for group in groups:
        nd_arg = {"z2xz2": 4, "z2_rc": 2}.get(group, 2)
        rep = hardware_representation(group, nd_arg)
        n_data = int(round(np.log2(rep.dim)))
        model = Mo.CQFAModel(rep, n_layers=max(1, n_layers))
        # inputs are fixed across seeds (deterministic); noise/sampling vary by seed
        input_rng = np.random.default_rng(1234)
        inputs = _inputs_for_group(group, input_rng)
        # theta
        if use_theta and n_layers > 0:
            theta = np.random.default_rng(99).standard_normal(model.n_params) * 0.3
            theta_id = 0
        else:
            theta = None
            theta_id = -1  # W=I
        eff_layers = n_layers if theta is not None else 0

        # observable: per-group Pauli with a nonzero symmetry-protected value
        observable = GROUP_OBSERVABLE.get(group, {0: "Z"})
        obs_sites = sorted(observable.keys())
        M = pauli_observable(observable, n_data)
        obs_label = "".join(observable[s] + str(s) for s in obs_sites)

        # backend kinds to run
        backend_specs = [("aer_noiseless", 0.0)]
        backend_specs += [(f"aer_depol:{p}", p) for p in depol_ps]
        if use_fake:
            backend_specs.append((f"fake:{fake_for[group]}", -1.0))

        for input_id, h, psi in inputs:
            m_circ, s_circ, _, n_frame = _prepared_circuits(
                rep, psi, theta, eff_layers, observable)
            # ground truth from statevector using the numpy observable M
            f_exact, p_exact = statevector_readout(
                Statevector.from_instruction(s_circ), n_data, n_frame, M=M)
            if f_exact is None:
                continue
            for kind, strength in backend_specs:
                for shots in shots_list:
                    for seed in range(n_seeds):
                        b = get_backend(kind, seed=1000 * seed + 7)
                        # skip fake if the circuit has too many qubits for it
                        try:
                            depth2q = b.transpiled_2q_depth(m_circ, seed=seed)
                            counts = b.run_sampler(m_circ, shots=shots, seed=seed)
                        except Exception as e:
                            if verbose:
                                print(f"[noise] skip {b.label} {group}/{input_id}: {e}")
                            break
                        r = readout_from_counts(counts, n_data, n_frame,
                                                observable_sites=obs_sites)
                        fp = r["f_hat_postsel"]
                        fr_ = r["f_hat_raw"]
                        rows.append(dict(
                            group=group, rep_name=rep.name, n_data=n_data,
                            n_frame=n_frame, backend_label=b.label,
                            noise_model=b.noise_model, noise_strength=strength,
                            input_id=input_id, h=h, observable=obs_label,
                            theta_id=theta_id, shots=shots,
                            seed=seed, transpiled_2q_depth=depth2q,
                            f_A_exact=f_exact, f_hat_postsel=fp, f_hat_raw=fr_,
                            abs_err_postsel=abs(fp - f_exact) if not np.isnan(fp) else float("nan"),
                            abs_err_raw=abs(fr_ - f_exact),
                            accept_rate=r["accept_rate"], p_exact=p_exact,
                            n_accepted=r["n_accepted"],
                        ))
            if verbose:
                print(f"[noise] {group}/{input_id}: f_exact={f_exact:.3f} "
                      f"p_exact={p_exact:.3f} done")

    summary = aggregate(rows)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        write_csv(os.path.join(out_dir, "noise_study.csv"), rows, NOISE_STUDY_FIELDS)
        write_csv(os.path.join(out_dir, "noise_study_summary.csv"), summary, SUMMARY_FIELDS)
    return rows, summary


# ---------------------------------------------------------------------------
# aggregation: mean +- std + bootstrap CI over seeds
# ---------------------------------------------------------------------------
def aggregate(rows):
    """Group by (group, backend_label, noise_strength, shots); mean/std/CI."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        key = (r["group"], r["backend_label"], r["noise_strength"], r["shots"])
        buckets[key].append(r)
    summary = []
    for (group, label, strength, shots), rs in sorted(
            buckets.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][3])):
        ep = np.array([x["abs_err_postsel"] for x in rs], dtype=float)
        er = np.array([x["abs_err_raw"] for x in rs], dtype=float)
        acc = np.array([x["accept_rate"] for x in rs], dtype=float)
        nacc = np.array([x["n_accepted"] for x in rs], dtype=float)
        lo_p, hi_p = bootstrap_ci(ep, seed=0)
        lo_r, hi_r = bootstrap_ci(er, seed=0)
        n_seeds = len(set(x["seed"] for x in rs))
        n_inputs = len(set(x["input_id"] for x in rs))
        summary.append(dict(
            group=group, backend_label=label, noise_strength=strength, shots=shots,
            n_runs=len(rs), n_seeds=n_seeds, n_inputs=n_inputs,
            mean_abs_err_postsel=float(np.nanmean(ep)),
            std_abs_err_postsel=float(np.nanstd(ep)),
            ci_lo_postsel=lo_p, ci_hi_postsel=hi_p,
            mean_abs_err_raw=float(np.nanmean(er)),
            std_abs_err_raw=float(np.nanstd(er)),
            ci_lo_raw=lo_r, ci_hi_raw=hi_r,
            mean_accept_rate=float(np.nanmean(acc)),
            mean_n_accepted=float(np.nanmean(nacc)),
        ))
    return summary


if __name__ == "__main__":
    run_noise_study(groups=("z2_rc",), shots_list=(256, 1024), n_seeds=2,
                    depol_ps=(1e-2, 1e-1), use_fake=False)
