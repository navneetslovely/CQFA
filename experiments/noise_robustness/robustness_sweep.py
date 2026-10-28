"""
experiments/robustness_sweep.py
===============================
Comprehensive noise robustness : sweep the two headline
advantages through ALL 10 noise models in experiments/noise_lib.NOISE_MODELS,
at several strengths, so the claim is "robust against noise of every standard
kind", not one or two channels.

  T_2copy  : the two-copy purity separation (coherent copies vs single-copy),
             each copy passed through the noise channel i.i.d. on all qubits.
  T_modeF  : the Mode-F hidden-shift accuracy, with the noise channel applied to
             every frame qubit (Hadamard-test operational features, finite shots).

Resume-aware (per model x strength parts) so it runs on the CPU fleet and can be
relaunched freely. Output: results/robustness/{twocopy,modef}_allnoise.csv .
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from experiments.noise_robustness import noise_lib as NL


# ---------------------------------------------------------------------------
# T_2copy: purity separation under an arbitrary channel
# ---------------------------------------------------------------------------
def _twocopy_unit(model, strength, n, m, seeds):
    from experiments.two_copy_separation.learning_separation import (
        make_dataset, coherent_learner, shadow_learner, copies_to_accuracy, purity)
    kraus = NL.NOISE_MODELS[model](strength)
    coh_b = [2, 4, 8, 16, 32, 64, 128]
    sc_b = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    coh, sc = [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        data = make_dataset(n, m, rng)
        noisy = []
        for rho, pur, y in data:
            rn = NL.apply_iid(rho, n, kraus)             # channel i.i.d. on all n qubits
            noisy.append((rn, purity(rn), y))
        c, _ = copies_to_accuracy(noisy, coherent_learner, coh_b, rng)
        s, _ = copies_to_accuracy(noisy, shadow_learner, sc_b, rng)
        coh.append(c if c else np.nan); sc.append(s if s else np.nan)
    cm = float(np.nanmedian(coh)); sm = float(np.nanmedian(sc)) if not np.all(np.isnan(sc)) else np.nan
    return dict(task="twocopy", model=model, strength=strength, n=n,
                coherent_copies=cm, singlecopy_copies=sm,
                separation_ratio=(sm / cm if cm and not np.isnan(sm) else np.nan))


# ---------------------------------------------------------------------------
# T_modeF: hidden-shift accuracy with a channel on every frame qubit
# ---------------------------------------------------------------------------
def _modef_unit(model, strength, N, m, seeds, shots=500):
    from experiments.fisher_information.fourier_mode import (
        _rand_state, _nearest_centroid_acc, hidden_shift_dataset, shift)
    from experiments.fisher_information.noise_resilience import (
        _joint_state, _fourier_frame_povm, sample_features)
    kraus = NL.NOISE_MODELS[model](strength)
    nfq = int(np.log2(N))
    accs = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        phi0, data = hidden_shift_dataset(N, m, rng)
        d = N
        X = []
        for psi, s, y in data:
            # ancilla-coherence Hadamard-test construction (as in noise_resilience),
            # with the channel applied to each frame qubit of both branches.
            phi_r = _joint_state(phi0, N); phi_p = _joint_state(psi, N)
            D = N * d
            Psi = np.zeros(2 * D, dtype=complex)
            Psi[:D] = phi_r / np.sqrt(2); Psi[D:] = phi_p / np.sqrt(2)
            rho = np.outer(Psi, Psi.conj())
            blocks = [rho[:D, :D], rho[:D, D:], rho[D:, :D], rho[D:, D:]]
            # apply channel to each frame qubit (frame is the high log2(N) qubits of
            # the (frame (x) data) factor); build embedded Kraus once.
            def chan(mat):
                for q in range(nfq):
                    ops = [NL.I2] * nfq
                    ops[q] = None
                    # frame-qubit Kraus embedded as (K on frame qubit q) (x) I_data
                    Fk = []
                    for K in kraus:
                        e = np.array([[1.0 + 0j]])
                        for qq in range(nfq):
                            e = np.kron(e, K if qq == q else NL.I2)
                        Fk.append(np.kron(e, np.eye(d)))
                    mat = sum(Kk @ mat @ Kk.conj().T for Kk in Fk)
                return mat
            r00, r01, r10, r11 = (chan(b) for b in blocks)
            chis = _fourier_frame_povm(N)
            probs = np.zeros(N); overlaps = np.zeros(N, dtype=complex)
            for k, chi in enumerate(chis):
                Pk = np.kron(chi.reshape(-1, 1), np.eye(d))
                probs[k] = float(np.real(np.trace(Pk.conj().T @ r11 @ Pk))) * 2.0
                overlaps[k] = 2.0 * np.trace(Pk.conj().T @ r10 @ Pk)
            X.append(sample_features(probs, overlaps, shots, rng))
        y = [y for _, _, y in data]
        accs.append(_nearest_centroid_acc(X, y))
    return dict(task="modef", model=model, strength=strength, N=N,
                acc=float(np.mean(accs)), acc_std=float(np.std(accs)))


# ---------------------------------------------------------------------------
# driver (resume-aware, parallel)
# ---------------------------------------------------------------------------
def _work_unit(kind, mdl, s, parts, n, N, m, seeds):
    """Top-level (picklable) worker: one (model, strength) unit, resume-aware."""
    safe = mdl.replace('/', '_').replace('(', '').replace(')', '').replace('=', '')
    part = os.path.join(parts, f"{safe}_{s:.3f}.json")
    if os.path.exists(part):
        return json.load(open(part))
    row = (_twocopy_unit(mdl, s, n, m, seeds) if kind == "twocopy"
           else _modef_unit(mdl, s, N, m, seeds))
    json.dump(row, open(part, "w"))
    return row


def _run(kind, out_dir, strengths, seeds, jobs, n=None, N=None, m=20):
    models = list(NL.NOISE_MODELS)
    units = [(mdl, s) for mdl in models for s in strengths]
    parts = os.path.join(out_dir, f"{kind}_parts")
    os.makedirs(parts, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_work_unit, kind, mdl, s, parts, n, N, m, seeds): (mdl, s)
                for mdl, s in units}
        for fut in as_completed(futs):
            rows.append(fut.result())
    rows.sort(key=lambda r: (r["model"], r["strength"]))
    pth = os.path.join(out_dir, f"{kind}_allnoise.csv")
    with open(pth, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[{kind}] {len(rows)} (model x strength) rows -> {pth}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["twocopy", "modef", "both"], default="both")
    ap.add_argument("--n", type=int, default=3, help="qubits for twocopy purity")
    ap.add_argument("--N", type=int, default=8, help="|G| for Mode F")
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--strengths", type=float, nargs="+",
                    default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", default="results/robustness")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if args.kind in ("twocopy", "both"):
        _run("twocopy", args.out, args.strengths, args.seeds, args.jobs, n=args.n, m=args.m)
    if args.kind in ("modef", "both"):
        _run("modef", args.out, args.strengths, args.seeds, args.jobs, N=args.N, m=args.m)


if __name__ == "__main__":
    main()
