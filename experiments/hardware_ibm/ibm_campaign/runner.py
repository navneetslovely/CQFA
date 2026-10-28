"""
experiments.hardware_ibm.ibm_campaign.runner
=========================================
Batched execution of the campaign PubSpecs, with a hard QPU-seconds gate and
full provenance capture. Three execution paths share ONE code path so the
offline dry run exercises exactly what the device run will do:

    run_offline(specs, "aer_noiseless")   ideal sampler   (correctness gate)
    run_offline(specs, "fake:fez")        Heron noise model (legibility check)
    run_ibm(specs, args)                  REAL device, three locks + budget gate

Every path returns a uniform result dict:
    {counts_by_id, transpiled_2q_depth_by_id, backend_label, job_id,
     usage, calibration, cost_estimate}

Cost model (see the QPU-budget analysis): quantum_seconds ~= JOB_OVERHEAD +
PER_SHOT * sum(shots). The true figure comes from ``job.usage_estimation`` /
``job.usage()`` on the device and is recorded verbatim; this estimate only
drives the pre-submit budget gate.

The API token is NEVER read or written here — auth is resolved by
``QiskitRuntimeService()`` from the user's saved config / env, and nothing from
the environment is copied into any artifact.
"""
from __future__ import annotations

from ..backends import get_backend, _join_register_counts


PER_SHOT_SECONDS = 0.00035          # shallow-circuit Heron per-shot execution time
JOB_OVERHEAD_SECONDS = 2.0          # amortized once for a single batched job


# ---------------------------------------------------------------------------
# cost estimate + budget gate
# ---------------------------------------------------------------------------
def estimate_cost(specs, per_shot=PER_SHOT_SECONDS, overhead=JOB_OVERHEAD_SECONDS):
    """Estimated batched-job cost. Returns a dict (also used by the CLI plan)."""
    total_shots = sum(s.total_shots() for s in specs)
    by_exp = {}
    for s in specs:
        e = by_exp.setdefault(s.experiment, {"n_pubs": 0, "shots": 0})
        e["n_pubs"] += 1
        e["shots"] += s.total_shots()
    return {
        "n_pubs": len(specs),
        "total_shots": total_shots,
        "est_qpu_seconds": round(overhead + per_shot * total_shots, 2),
        "per_shot_seconds": per_shot,
        "overhead_seconds": overhead,
        "by_experiment": by_exp,
    }


class BudgetExceeded(RuntimeError):
    pass


def enforce_budget(specs, budget_seconds):
    cost = estimate_cost(specs)
    if cost["est_qpu_seconds"] > budget_seconds:
        raise BudgetExceeded(
            f"estimated {cost['est_qpu_seconds']}s > budget {budget_seconds}s "
            f"({cost['total_shots']} shots over {cost['n_pubs']} PUBs)")
    return cost


# ---------------------------------------------------------------------------
# shared PUB execution (counts parsing identical for Aer / Fake / IBM)
# ---------------------------------------------------------------------------
def _pub_counts(pub_result) -> dict:
    """Joined per-register counts dict for one SamplerV2 PUB result."""
    data = pub_result.data
    reg_names = list(data.keys()) if hasattr(data, "keys") else None
    if reg_names is None:
        return pub_result.data.meas.get_counts()
    return _join_register_counts(data, reg_names)


def _run_pubs(sampler, circuits_with_shots):
    """Run [(circuit, shots), ...] as ONE job of per-PUB-shots PUBs."""
    pubs = [(circ, None, int(shots)) for circ, shots in circuits_with_shots]
    job = sampler.run(pubs)
    result = job.result()
    return job, result


# ---------------------------------------------------------------------------
# offline path (Aer ideal / Fake device-noise model)
# ---------------------------------------------------------------------------
def run_offline(specs, backend_kind="aer_noiseless", seed=1234):
    """Run the whole campaign on an offline backend as ONE per-PUB-shots job."""
    b = get_backend(backend_kind, seed=seed)
    tcircs = [b.transpile(s.circuit, seed=seed) for s in specs]
    tcircs = [c[0] if isinstance(c, list) else c for c in tcircs]
    depth_by_id = {}
    for s, tc in zip(specs, tcircs):
        try:
            depth_by_id[s.pub_id] = int(
                tc.depth(lambda instr: instr.operation.num_qubits == 2))
        except Exception:
            depth_by_id[s.pub_id] = int(tc.depth())
    job, result = _run_pubs(b._sampler, list(zip(tcircs, [s.shots for s in specs])))
    counts_by_id = {s.pub_id: _pub_counts(result[i]) for i, s in enumerate(specs)}
    return {
        "counts_by_id": counts_by_id,
        "transpiled_2q_depth_by_id": depth_by_id,
        "transpiled_circuits": list(tcircs),
        "final_layout_by_id": {s.pub_id: None for s in specs},   # no physical layout offline
        "backend_label": b.label,
        "job_id": None,
        "usage": None,
        "calibration": {"kind": "offline", "backend": b.label},
        "cost_estimate": estimate_cost(specs),
    }


# ---------------------------------------------------------------------------
# gated real-device path
# ---------------------------------------------------------------------------
def _calibration_snapshot(backend, used_qubits, used_pairs=(), source="layout"):
    """Compact, reproducible calibration snapshot for the used qubits AND the used
    two-qubit edges (the dominant noise source; C7 is a 2q-depth experiment). Never
    the token. Best-effort: wrapped so a missing field never aborts the run."""
    snap = {"backend": getattr(backend, "name", "unknown"),
            "used_qubits": sorted(int(q) for q in used_qubits),
            "used_qubits_source": source}
    try:
        snap["num_qubits"] = int(backend.num_qubits)
        snap["basis_gates"] = list(getattr(backend, "operation_names", []))
    except Exception:
        pass
    try:
        tgt = backend.target
        props = {}
        for q in sorted(used_qubits):
            qp = {}
            for attr in ("T1", "T2", "frequency"):
                try:
                    qp[attr] = float(getattr(tgt.qubit_properties(q), attr.lower()))
                except Exception:
                    pass
            # single-qubit readout + sx error where exposed
            for gate in ("sx", "x", "measure"):
                try:
                    ip = tgt[gate].get((q,))
                    if ip is not None and ip.error is not None:
                        qp[f"{gate}_error"] = float(ip.error)
                    if ip is not None and getattr(ip, "duration", None) is not None:
                        qp[f"{gate}_duration"] = float(ip.duration)
                except Exception:
                    pass
            props[str(q)] = qp
        snap["qubit_properties"] = props
    except Exception:
        pass
    # two-qubit gate errors on the edges actually used (ecr/cz/cx, either ordering)
    try:
        tgt = backend.target
        pairs = {}
        for (a, b) in sorted(used_pairs):
            entry = {}
            for gname in ("ecr", "cz", "cx"):
                try:
                    ip = tgt[gname].get((a, b)) or tgt[gname].get((b, a))
                except Exception:
                    ip = None
                if ip is not None:
                    if getattr(ip, "error", None) is not None:
                        entry[f"{gname}_error"] = float(ip.error)
                    if getattr(ip, "duration", None) is not None:
                        entry[f"{gname}_duration"] = float(ip.duration)
            if entry:
                pairs[f"{a}_{b}"] = entry
        snap["two_qubit_gates"] = pairs
    except Exception:
        pass
    return snap


def run_ibm(specs, args, service=None):
    """Submit the campaign as ONE batched job to the least-busy real device.

    Requires three locks (CQFA_IBM_RUN=1, --ibm, --ibm-confirm) via
    ``ibm_runtime_path`` AND the estimated QPU-seconds to fit ``args.budget_seconds``.
    Raises before any submission if either fails. Never touches the token.
    """
    from ..ibm_runtime_path import _check_locks
    _check_locks(args)                                   # raises IBMDisabled if not fully unlocked
    cost = enforce_budget(specs, getattr(args, "budget_seconds", 300.0))

    # belt-and-suspenders hard caps (mirror ibm_runtime_path.run_ibm)
    max_shots = int(getattr(args, "max_shots", 8192))
    max_circuits = int(getattr(args, "max_circuits", 400))
    max_total = int(getattr(args, "max_total_shots", 1_000_000))
    assert len(specs) <= max_circuits, f"cap: {len(specs)} PUBs > max_circuits={max_circuits}"
    assert all(s.shots <= max_shots for s in specs), f"cap: a PUB exceeds max_shots={max_shots}"
    assert cost["total_shots"] <= max_total, f"cap: {cost['total_shots']} > max_total_shots={max_total}"

    if service is None:                                  # pragma: no cover (needs creds)
        import os
        from qiskit_ibm_runtime import QiskitRuntimeService
        token = os.environ.get("QISKIT_IBM_TOKEN")
        if token:
            # in-memory auth from env (token NEVER saved to ~/.qiskit or any artifact)
            kw = {"channel": os.environ.get("QISKIT_IBM_CHANNEL", "ibm_quantum_platform"),
                  "token": token}
            inst = os.environ.get("QISKIT_IBM_INSTANCE")
            if inst:
                kw["instance"] = inst
            service = QiskitRuntimeService(**kw)
        else:
            service = QiskitRuntimeService()             # fall back to a saved account
    min_q = max((s.circuit.num_qubits for s in specs), default=3)
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=min_q)

    from qiskit import transpile
    tcircs = transpile([s.circuit for s in specs], backend, optimization_level=1)
    depth_by_id, used_qubits, used_pairs = {}, set(), set()
    final_layout_by_id, layout_fail = {}, 0
    for s, tc in zip(specs, tcircs):
        try:
            depth_by_id[s.pub_id] = int(
                tc.depth(lambda instr: instr.operation.num_qubits == 2))
        except Exception:
            depth_by_id[s.pub_id] = int(tc.depth())
        # physical qubits actually mapped (best-effort; record per-PUB layout)
        try:
            layout = [int(q) for q in tc.layout.final_index_layout()]
            final_layout_by_id[s.pub_id] = layout
            used_qubits.update(layout)
        except Exception:
            final_layout_by_id[s.pub_id] = None
            layout_fail += 1
        # 2q edges actually used, for the calibration snapshot
        try:
            for instr in tc.data:
                if instr.operation.num_qubits == 2:
                    qs = tuple(sorted(tc.find_bit(b).index for b in instr.qubits))
                    used_pairs.add(qs)
        except Exception:
            pass
    cal_source = "layout" if used_qubits else "FALLBACK-range"
    if not used_qubits:
        print(f"[warn] layout extraction failed for all {layout_fail} circuits; "
              f"calibration falls back to physical qubits {sorted(range(min_q))}, "
              f"which are almost certainly NOT the mapped qubits — treat "
              f"calibration.used_qubits_source='FALLBACK-range' as unreliable.")

    print(f"[IBM] backend={backend.name} pubs={len(specs)} "
          f"total_shots={cost['total_shots']} est_qpu_s={cost['est_qpu_seconds']} "
          f"(budget {getattr(args, 'budget_seconds', 300.0)}s). Submitting ONE batched job.")

    from qiskit_ibm_runtime import Batch, SamplerV2
    with Batch(backend=backend) as batch:
        sampler = SamplerV2(mode=batch)
        job, result = _run_pubs(
            sampler, list(zip(tcircs, [s.shots for s in specs])))
    job_id = job.job_id()

    usage = {}
    for meth in ("usage_estimation", "usage"):
        try:
            u = getattr(job, meth)
            usage[meth] = u() if callable(u) else u
        except Exception:
            pass
    try:
        usage["metrics"] = job.metrics()
    except Exception:
        pass

    counts_by_id = {s.pub_id: _pub_counts(result[i]) for i, s in enumerate(specs)}
    return {
        "counts_by_id": counts_by_id,
        "transpiled_2q_depth_by_id": depth_by_id,
        "transpiled_circuits": list(tcircs),
        "final_layout_by_id": final_layout_by_id,
        "backend_label": f"IBM:{backend.name}",
        "job_id": job_id,
        "usage": usage,
        "calibration": _calibration_snapshot(
            backend, used_qubits or set(range(min_q)), used_pairs, source=cal_source),
        "cost_estimate": cost,
    }
