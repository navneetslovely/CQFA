
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from cqfa import readout as Mo, frame as L, simulator as S
from .cqfa_circuit import (
    build_cqfa_circuit, hardware_representation, numpy_to_qiskit_amplitudes,
    qiskit_to_numpy_amplitudes,
)
from .readout import statevector_readout, hadamard_from_counts


# ---------------------------------------------------------------------------
# statevector helpers
# ---------------------------------------------------------------------------
def _init_circuit(psi_q, n_data, n_frame):
    d = QuantumRegister(n_data, "data")
    fr = QuantumRegister(n_frame, "frame")
    qc = QuantumCircuit(d, fr)
    qc.initialize(psi_q, [d[i] for i in range(n_data)])
    return qc


def qiskit_statevector_feature(rep, psi, theta, M=None):
 
    n_data = int(round(np.log2(rep.dim)))
    n_frame = rep.n_frame_qubits
    n_layers = 0 if theta is None else len(np.asarray(theta).ravel()) // (n_data * 2)
    qc, _ = build_cqfa_circuit(rep, theta=theta, mode="postselect",
                               measure=False, n_layers=n_layers)
    psi_q = numpy_to_qiskit_amplitudes(psi, n_data)
    full = _init_circuit(psi_q, n_data, n_frame).compose(qc)
    sv = Statevector.from_instruction(full)
    return statevector_readout(sv, n_data, n_frame, M=M)


# ---------------------------------------------------------------------------
# per-rep equivalence sweep
# ---------------------------------------------------------------------------
def check_rep(rep_name, n_data=None, n_trials=20, n_thetas=5, n_layers=3, seed=0,
              tol=1e-9):

    rep = hardware_representation(rep_name, n_data)
    nd = int(round(np.log2(rep.dim)))
    model = Mo.CQFAModel(rep, n_layers=n_layers)
    O_z0 = S.local_z(nd, 0)
    rng = np.random.default_rng(seed)
    rows = []
    worst_f = 0.0
    worst_p = 0.0
    thetas = [rng.standard_normal(model.n_params) for _ in range(n_thetas)]
    thetas[0] = np.zeros(model.n_params)     # include W = I (reference case)
    for ti, theta in enumerate(thetas):
        for tr in range(n_trials):
            psi = rng.standard_normal(rep.dim) + 1j * rng.standard_normal(rep.dim)
            psi /= np.linalg.norm(psi)

            # (1) Z_0 vs CQFAModel.feature (the headline gate)
            f_np = model.feature(theta, psi)[0]
            p_np = L.symmetric_fraction(psi, rep)
            f_qk, p_qk = qiskit_statevector_feature(rep, psi, theta, M=None)
            if f_qk is None:
                continue
            df = abs(f_np - f_qk)
            dp = abs(p_np - p_qk)
            worst_f = max(worst_f, df)
            worst_p = max(worst_p, dp)
            rows.append(dict(
                group=rep_name, rep_name=rep.name, n_data=nd,
                n_frame=rep.n_frame_qubits, readout="postselect", trial=tr,
                observable="Z_site0", theta_id=ti,
                f_A_qiskit=f_qk, f_A_numpy=f_np, abs_diff_f=df,
                p_qiskit=p_qk, p_numpy=p_np, abs_diff_p=dp,
                **{"pass": bool(df < tol and dp < tol)},
            ))

            # (2) random Hermitian M, W = I, direct numpy post-selected readout
            if ti == 0:
                Mm = rng.standard_normal((rep.dim, rep.dim)) + 1j * rng.standard_normal((rep.dim, rep.dim))
                M = (Mm + Mm.conj().T) / 2
                v_np, _ = L.cqfa_readout(psi, rep, M)
                # qiskit statevector with W=I, custom M
                f_qk_m, _ = qiskit_statevector_feature(rep, psi, None, M=M)
                if v_np is not None and f_qk_m is not None:
                    dfm = abs(v_np - f_qk_m)
                    worst_f = max(worst_f, dfm)
                    rows.append(dict(
                        group=rep_name, rep_name=rep.name, n_data=nd,
                        n_frame=rep.n_frame_qubits, readout="postselect", trial=tr,
                        observable="random_hermitian", theta_id=-1,
                        f_A_qiskit=f_qk_m, f_A_numpy=v_np, abs_diff_f=dfm,
                        p_qiskit=p_qk, p_numpy=p_np, abs_diff_p=dp,
                        **{"pass": bool(dfm < tol)},
                    ))
    return rows, max(worst_f, worst_p)


# ---------------------------------------------------------------------------
# Hadamard-test diagnostic cross-check
# ---------------------------------------------------------------------------
def check_hadamard(rep_name, n_data=None, n_trials=10, seed=0, tol=1e-9):
    """Statevector Hadamard test == Re<psi|A|psi> for the product involution A.

    Returns worst abs diff.
    """
    rep = hardware_representation(rep_name, n_data)
    nd = int(round(np.log2(rep.dim)))
    rng = np.random.default_rng(seed)
    # A = product of the non-identity generators (the involution the Hadamard
    # test reads). For z2 reps this is matrices[1]; for z2xz2 it is P_even*P_odd.
    if rep.name.lower().startswith("z2xz2"):
        A = rep.matrices[1] @ rep.matrices[2]
    else:
        A = rep.matrices[1]
    worst = 0.0
    from qiskit.quantum_info import Statevector as SV
    for _ in range(n_trials):
        psi = rng.standard_normal(rep.dim) + 1j * rng.standard_normal(rep.dim)
        psi /= np.linalg.norm(psi)
        re_np = float(np.real(psi.conj() @ A @ psi))
        # build hadamard circuit (unmeasured) + statevector <Z_anc>
        qc, rmap = build_cqfa_circuit(rep, theta=None, mode="hadamard", measure=False)
        psi_q = numpy_to_qiskit_amplitudes(psi, nd)
        anc = QuantumRegister(1, "anc"); d = QuantumRegister(nd, "data")
        initc = QuantumCircuit(anc, d)
        initc.initialize(psi_q, [d[i] for i in range(nd)])
        full = initc.compose(qc)
        sv = SV.from_instruction(full).data
        # <Z_anc>: ancilla is qubit 0 (LSB). P(anc=0)-P(anc=1).
        p0 = sum(abs(sv[i]) ** 2 for i in range(len(sv)) if (i & 1) == 0)
        p1 = 1.0 - p0
        z_anc = p0 - p1
        worst = max(worst, abs(z_anc - re_np))
    return worst


# ---------------------------------------------------------------------------
# top-level runner (used by CLI and tests)
# ---------------------------------------------------------------------------
DEFAULT_REPS = [("z2_rc", 2), ("z2_pauli", 2), ("z2_pauli", 3), ("z2xz2", 4)]


def run_equivalence(reps=None, n_trials=20, n_thetas=5, n_layers=3, seed=0,
                    tol=1e-9, verbose=True):
    """Run the full equivalence sweep. Returns (all_rows, overall_worst, ok)."""
    reps = reps if reps is not None else DEFAULT_REPS
    all_rows = []
    overall_worst = 0.0
    for rep_name, nd in reps:
        rows, worst = check_rep(rep_name, nd, n_trials=n_trials, n_thetas=n_thetas,
                                n_layers=n_layers, seed=seed, tol=tol)
        had = check_hadamard(rep_name, nd, seed=seed, tol=tol)
        worst = max(worst, had)
        overall_worst = max(overall_worst, worst)
        all_rows.extend(rows)
        if verbose:
            print(f"[equivalence] {rep_name}(n={nd}): worst|diff|={worst:.2e} "
                  f"(hadamard {had:.2e})  {'PASS' if worst < tol else 'FAIL'}")
    ok = overall_worst < tol
    if verbose:
        print(f"[equivalence] OVERALL worst={overall_worst:.2e}  "
              f"{'PASS' if ok else 'FAIL'} (tol={tol:g})")
    return all_rows, overall_worst, ok


if __name__ == "__main__":
    run_equivalence()
