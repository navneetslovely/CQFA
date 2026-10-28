
from __future__ import annotations

import argparse
import os
import sys

from ._util import write_csv, write_manifest
from . import equivalence as eq
from . import noise_study as ns


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "out")


def _parse_list(s, cast):
    return [cast(x) for x in str(s).split(",") if x != ""]


def build_parser():
    p = argparse.ArgumentParser(description="CQFA hardware experiment driver")
    p.add_argument("--task", choices=["equivalence", "noise", "ibm"],
                   default="equivalence")
    p.add_argument("--group", default="z2_rc,z2xz2",
                   help="comma list of groups for the noise task")
    p.add_argument("--n-data", type=int, default=None)
    p.add_argument("--readout", choices=["postselect", "hadamard"],
                   default="postselect")
    p.add_argument("--shots", default="256,1024,4096,16384")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--depol-p", default="1e-3,3e-3,1e-2,3e-2,1e-1")
    p.add_argument("--n-layers", type=int, default=0,
                   help="ansatz layers (0 -> W=I)")
    p.add_argument("--no-fake", action="store_true",
                   help="skip FakeBackend arms (simulators only)")
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--out", default=DEFAULT_OUT)
    # IBM locks + budget caps (all off by default)
    p.add_argument("--ibm", action="store_true")
    p.add_argument("--ibm-confirm", action="store_true")
    p.add_argument("--max-shots", type=int, default=1024)
    p.add_argument("--max-circuits", type=int, default=4)
    p.add_argument("--max-total-shots", type=int, default=4096)
    return p


def task_equivalence(args) -> int:
    reps = None
    if args.group and args.group != "z2_rc,z2xz2":
        reps = [(g, args.n_data) for g in args.group.split(",")]
    rows, worst, ok = eq.run_equivalence(reps=reps, tol=args.tol, verbose=True)
    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "equivalence.csv"), rows)
    write_manifest(args.out, sys.argv, seeds=[0],
                   extra={"task": "equivalence", "worst_abs_diff": worst,
                          "pass": ok, "tol": args.tol})
    print(f"[run_hardware] equivalence -> {os.path.join(args.out, 'equivalence.csv')} "
          f"({'PASS' if ok else 'FAIL'})")
    return 0 if ok else 2


def task_noise(args) -> int:
    groups = tuple(args.group.split(","))
    depol_ps = tuple(_parse_list(args.depol_p, float))
    shots_list = tuple(_parse_list(args.shots, int))
    rows, summary = ns.run_noise_study(
        groups=groups, depol_ps=depol_ps, shots_list=shots_list,
        n_seeds=args.seeds, n_layers=args.n_layers,
        use_fake=not args.no_fake, use_theta=(args.n_layers > 0),
        out_dir=args.out, verbose=True)
    write_manifest(args.out, sys.argv, seeds=list(range(args.seeds)),
                   extra={"task": "noise", "n_rows": len(rows)})
    print(f"[run_hardware] noise -> {os.path.join(args.out, 'noise_study.csv')} "
          f"({len(rows)} rows, {len(summary)} summary rows)")
    return 0


def task_ibm(args) -> int:
    """Attempt the gated IBM path; fall back to FakeBackend otherwise."""
    import json
    import numpy as np
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.quantum_info import Statevector
    from . import ibm_runtime_path as ibmp
    from .cqfa_circuit import build_cqfa_circuit, hardware_representation, pauli_observable
    from .backends import get_backend
    from .readout import readout_from_counts, statevector_readout

    group = args.group.split(",")[0]
    nd = args.n_data if args.n_data is not None else (4 if group == "z2xz2" else 2)
    rep = hardware_representation(group, nd)
    n_data = int(round(np.log2(rep.dim)))
    n_frame = rep.n_frame_qubits
    observable = ns.GROUP_OBSERVABLE.get(group, {0: "Z"})
    obs_sites = sorted(observable.keys())
    M = pauli_observable(observable, n_data)

    # shallow, hardware-friendly prep (physical for the device arm)
    if group == "z2xz2":
        prep = ibmp.shallow_cluster_prep(n_data)
    else:
        prep = ibmp.shallow_random_prep(n_data, seed=0)
    core, _ = build_cqfa_circuit(rep, theta=None, mode="postselect",
                                 measure=True, n_layers=0, observable=observable)
    d = QuantumRegister(n_data, "data"); fr = QuantumRegister(n_frame, "frame")
    base = QuantumCircuit(d, fr)
    base = base.compose(prep, qubits=[d[i] for i in range(n_data)])
    circ = base.compose(core)

    # ground truth (statevector) for reporting the abs error
    core_s, _ = build_cqfa_circuit(rep, theta=None, mode="postselect",
                                   measure=False, n_layers=0)
    base_s = QuantumCircuit(d, fr).compose(prep, qubits=[d[i] for i in range(n_data)])
    f_exact, p_exact = statevector_readout(
        Statevector.from_instruction(base_s.compose(core_s)), n_data, n_frame, M=M)

    os.makedirs(args.out, exist_ok=True)
    shots = min(args.max_shots, 1024)
    try:
        res = ibmp.run_ibm([circ], shots=shots, args=args, min_num_qubits=n_data + n_frame)
        counts = res["counts"]
        backend_label = res["backend_label"]
        job_id = res["job_id"]
        depth2q = res["transpiled_2q_depth"]
        with open(os.path.join(args.out, "ibm_manifest.json"), "w") as f:
            json.dump({"backend_label": backend_label, "job_id": job_id,
                       "shots": shots, "group": group,
                       "transpiled_2q_depth": depth2q}, f, indent=2)
    except ibmp.IBMDisabled as e:
        fake = "sherbrooke" if group == "z2xz2" else "manila"
        print(f"[IBM] real-hardware path DISABLED ({e}). "
              f"Using FakeBackend fallback Fake:{fake}.")
        b = get_backend(f"fake:{fake}", seed=0)
        counts = b.run_sampler(circ, shots=shots, seed=0)
        backend_label = b.label
        job_id = None

    r = readout_from_counts(counts, n_data, n_frame, observable_sites=obs_sites)
    row = dict(group=group, rep_name=rep.name, n_data=n_data, n_frame=n_frame,
               backend_label=backend_label, job_id=job_id, shots=shots,
               f_A_exact=f_exact, f_hat_postsel=r["f_hat_postsel"],
               f_hat_raw=r["f_hat_raw"],
               abs_err_postsel=abs(r["f_hat_postsel"] - f_exact),
               abs_err_raw=abs(r["f_hat_raw"] - f_exact),
               accept_rate=r["accept_rate"], p_exact=p_exact,
               n_accepted=r["n_accepted"])
    write_csv(os.path.join(args.out, "ibm_run.csv"), [row])
    write_manifest(args.out, sys.argv, seeds=[0],
                   extra={"task": "ibm", "backend_label": backend_label,
                          "job_id": job_id})
    print(f"[run_hardware] ibm -> {os.path.join(args.out, 'ibm_run.csv')} "
          f"(backend {backend_label}, f_postsel={r['f_hat_postsel']:.3f}, "
          f"f_exact={f_exact:.3f})")
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.task == "equivalence":
        return task_equivalence(args)
    if args.task == "noise":
        return task_noise(args)
    if args.task == "ibm":
        return task_ibm(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
