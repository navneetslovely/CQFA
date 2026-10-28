"""
experiments.hardware_ibm.ibm_runtime_path
======================================
Budget-guarded REAL-HARDWARE path via qiskit-ibm-runtime. OFF BY DEFAULT.

The author has ~10 minutes of IBM budget; this path must NEVER be spent
automatically. THREE independent locks are required before a single job is
submitted:

    1. env var  CQFA_IBM_RUN=1
    2. CLI flag --ibm         (args.ibm)
    3. CLI flag --ibm-confirm (args.ibm_confirm)

Plus hard budget caps: n_circuits * shots must be <= args.max_total_shots
(default 4096). Absent any lock, ``run_ibm`` raises and the caller
(``run_hardware.py``) falls back to the matching FakeBackend device-noise model.

No token is ever embedded; auth is read from the saved qiskit config or the
QISKIT_IBM_TOKEN env var by ``QiskitRuntimeService()``.
"""
from __future__ import annotations

import os


class IBMDisabled(RuntimeError):
    """Raised when the gated IBM path is not fully enabled (caller falls back)."""


def ibm_enabled(args) -> bool:
    """True iff all three locks are present. Does NOT submit anything."""
    return (os.environ.get("CQFA_IBM_RUN") == "1"
            and getattr(args, "ibm", False)
            and getattr(args, "ibm_confirm", False))


def _check_locks(args):
    if os.environ.get("CQFA_IBM_RUN") != "1":
        raise IBMDisabled("CQFA_IBM_RUN != 1")
    if not getattr(args, "ibm", False):
        raise IBMDisabled("--ibm not passed")
    if not getattr(args, "ibm_confirm", False):
        raise IBMDisabled("refuse to spend budget without --ibm-confirm")


def run_ibm(circuits, shots, args, min_num_qubits=3, service=None):
    """Submit ONE job to the least-busy real device, under three locks + caps.

    Parameters
    ----------
    circuits : list[QuantumCircuit]
        Already state-prepared, measured circuits (prefer shallow prep, see
        ``shallow_cluster_prep``).
    shots : int
        Shots per circuit (capped by args.max_shots).
    args : argparse.Namespace
        Must carry .ibm, .ibm_confirm, .max_shots, .max_circuits,
        .max_total_shots.
    service : QiskitRuntimeService or None
        Injected for testing (mock). If None, constructed from saved creds.

    Returns
    -------
    dict with keys: backend_label, job_id, counts (list), n_circuits, shots.

    Raises
    ------
    IBMDisabled if any lock is missing (caller should fall back to Fake).
    AssertionError if a budget cap is exceeded.
    """
    _check_locks(args)

    single = not isinstance(circuits, (list, tuple))
    circs = [circuits] if single else list(circuits)

    # -- budget caps (belt and suspenders) ----------------------------------
    max_shots = int(getattr(args, "max_shots", 1024))
    max_circuits = int(getattr(args, "max_circuits", 4))
    max_total = int(getattr(args, "max_total_shots", 4096))
    shots = min(int(shots), max_shots)
    assert len(circs) <= max_circuits, (
        f"budget cap: {len(circs)} circuits > max_circuits={max_circuits}")
    assert len(circs) * shots <= max_total, (
        f"budget cap: {len(circs)}*{shots} > max_total_shots={max_total}")

    # -- service + backend --------------------------------------------------
    if service is None:                       # pragma: no cover (needs real creds)
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, simulator=False,
                                 min_num_qubits=min_num_qubits)

    from qiskit import transpile
    tqc = transpile(circs, backend, optimization_level=1)
    depth2q = [c.depth(lambda instr: instr.operation.num_qubits == 2) for c in tqc]

    # -- estimated-time gate (printed; the third lock already guards spend) --
    print(f"[IBM] backend={backend.name} n_circuits={len(circs)} shots={shots} "
          f"transpiled_2q_depth={depth2q}")
    print(f"[IBM] estimated total shots = {len(circs) * shots} "
          f"(cap {max_total}). Submitting ONE batched job.")

    # -- single batched job -------------------------------------------------
    from qiskit_ibm_runtime import Batch, SamplerV2
    with Batch(backend=backend) as batch:
        sampler = SamplerV2(mode=batch)
        job = sampler.run(tqc, shots=shots)
        result = job.result()
    job_id = job.job_id()

    counts = []
    for pub in result:
        data = pub.data
        reg_names = list(data.keys()) if hasattr(data, "keys") else None
        if reg_names is None:
            counts.append(pub.data.meas.get_counts())
        else:
            from .backends import _join_register_counts
            counts.append(_join_register_counts(data, reg_names))

    return {
        "backend_label": f"IBM:{backend.name}",
        "job_id": job_id,
        "counts": counts if not single else counts[0],
        "n_circuits": len(circs),
        "shots": shots,
        "transpiled_2q_depth": depth2q,
    }


# ---------------------------------------------------------------------------
# shallow, hardware-friendly state prep (avoids deep initialize() on device)
# ---------------------------------------------------------------------------
def shallow_cluster_prep(n_data):
    """Cluster-state prep (H on all + CZ on the chain) — the physical SPT input.

    Returns a QuantumCircuit on n_data qubits. Transpiles shallowly (linear
    depth) unlike an arbitrary ``initialize``. Used for the Z2xZ2 IBM arm.
    """
    from qiskit import QuantumCircuit, QuantumRegister
    d = QuantumRegister(n_data, "data")
    qc = QuantumCircuit(d)
    for q in range(n_data):
        qc.h(d[q])
    for q in range(n_data - 1):
        qc.cz(d[q], d[q + 1])
    return qc


def shallow_random_prep(n_data, seed=0):
    """Shallow single-qubit-rotation product state (for the Z2 IBM arm)."""
    import numpy as np
    from qiskit import QuantumCircuit, QuantumRegister
    rng = np.random.default_rng(seed)
    d = QuantumRegister(n_data, "data")
    qc = QuantumCircuit(d)
    for q in range(n_data):
        qc.ry(float(rng.uniform(0, np.pi)), d[q])
        qc.rz(float(rng.uniform(0, 2 * np.pi)), d[q])
    return qc
