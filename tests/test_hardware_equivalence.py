
import types
import unittest

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from experiments.hardware_ibm import equivalence as eq
from experiments.hardware_ibm.cqfa_circuit import (
    build_cqfa_circuit, hardware_representation,
    numpy_to_qiskit_amplitudes, qiskit_to_numpy_amplitudes, pauli_observable,
)
from experiments.hardware_ibm.backends import get_backend
from experiments.hardware_ibm.readout import readout_from_counts, statevector_readout
from experiments.hardware_ibm import ibm_runtime_path as ibmp
from experiments.hardware_ibm import qiskit_cqfa_reference as ref
from cqfa import frame as L


TOL = 1e-9


class TestHardwareEquivalence(unittest.TestCase):
    # -- the < 1e-9 statevector gate, per group ----------------------------
    def test_z2_statevector_matches_numpy(self):
        rows, worst = eq.check_rep("z2_rc", 2, n_trials=15, n_thetas=4, seed=0)
        self.assertLess(worst, TOL, f"z2_rc worst diff {worst:.2e}")
        self.assertTrue(all(r["pass"] for r in rows))

    def test_z2pauli3_statevector_matches_numpy(self):
        rows, worst = eq.check_rep("z2_pauli", 3, n_trials=12, n_thetas=3, seed=1)
        self.assertLess(worst, TOL, f"z2_pauli3 worst diff {worst:.2e}")

    def test_z2xz2_statevector_matches_numpy(self):
        rows, worst = eq.check_rep("z2xz2", 4, n_trials=12, n_thetas=3, seed=2)
        self.assertLess(worst, TOL, f"z2xz2 worst diff {worst:.2e}")

    def test_hadamard_diagnostic_matches_numpy(self):
        for name, nd in (("z2_rc", 2), ("z2xz2", 4)):
            worst = eq.check_hadamard(name, nd, n_trials=10, seed=3)
            self.assertLess(worst, TOL, f"hadamard {name} {worst:.2e}")

    # -- qubit-ordering round trip -----------------------------------------
    def test_qubit_ordering_roundtrip(self):
        rng = np.random.default_rng(4)
        for n_data in (2, 3, 4):
            psi = rng.standard_normal(2 ** n_data) + 1j * rng.standard_normal(2 ** n_data)
            back = qiskit_to_numpy_amplitudes(
                numpy_to_qiskit_amplitudes(psi, n_data), n_data)
            self.assertTrue(np.allclose(psi, back, atol=1e-14))
            fwd = numpy_to_qiskit_amplitudes(
                qiskit_to_numpy_amplitudes(psi, n_data), n_data)
            self.assertTrue(np.allclose(psi, fwd, atol=1e-14))

    # -- the blessed Z2 reference still passes -----------------------------
    def test_reference_selftest_still_passes(self):
        self.assertTrue(ref._selftest(n_trials=30))

    # -- physical post-selection realises p = ||Pi psi||^2 (shot-based) ----
    def test_accept_rate_matches_p(self):
        rep = hardware_representation("z2_rc", 2)
        n_data, n_frame = 2, 1
        rng = np.random.default_rng(5)
        psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
        psi /= np.linalg.norm(psi)
        p_np = L.symmetric_fraction(psi, rep)
        qc, _ = build_cqfa_circuit(rep, theta=None, mode="postselect", measure=True)
        d = QuantumRegister(n_data, "data"); fr = QuantumRegister(n_frame, "frame")
        initc = QuantumCircuit(d, fr)
        initc.initialize(numpy_to_qiskit_amplitudes(psi, n_data),
                         [d[i] for i in range(n_data)])
        counts = get_backend("aer_noiseless", seed=7).run_sampler(
            initc.compose(qc), shots=8192)
        r = readout_from_counts(counts, n_data, n_frame, observable_site=0)
        self.assertAlmostEqual(r["accept_rate"], p_np, delta=0.05)

    # -- post-selection is no worse than raw at noiseless (direction guard) -
    def test_postsel_no_worse_than_raw_noiseless(self):
        rep = hardware_representation("z2_rc", 2)
        n_data, n_frame = 2, 1
        observable = {0: "X", 1: "Z"}                 # XZ: nonzero protected value
        obs_sites = sorted(observable.keys())
        M = pauli_observable(observable, n_data)
        # pick a state with a clearly nonzero f_A
        rng = np.random.default_rng(3)
        psi = None
        for _ in range(500):
            c = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            c /= np.linalg.norm(c)
            fA, p = L.cqfa_readout(c, rep, M)
            if fA is not None and fA > 0.8 and 0.3 < p < 0.7:
                psi = c
                break
        self.assertIsNotNone(psi)
        d = QuantumRegister(n_data, "data"); fr = QuantumRegister(n_frame, "frame")
        initc = QuantumCircuit(d, fr)
        initc.initialize(numpy_to_qiskit_amplitudes(psi, n_data),
                         [d[i] for i in range(n_data)])
        qc_m, _ = build_cqfa_circuit(rep, None, "postselect", measure=True,
                                     observable=observable)
        qc_s, _ = build_cqfa_circuit(rep, None, "postselect", measure=False)
        from qiskit.quantum_info import Statevector
        f_exact, _ = statevector_readout(
            Statevector.from_instruction(initc.compose(qc_s)), n_data, n_frame, M=M)
        counts = get_backend("aer_noiseless", seed=11).run_sampler(
            initc.compose(qc_m), shots=8192)
        r = readout_from_counts(counts, n_data, n_frame, observable_sites=obs_sites)
        err_ps = abs(r["f_hat_postsel"] - f_exact)
        err_raw = abs(r["f_hat_raw"] - f_exact)
        self.assertLessEqual(err_ps, err_raw + 0.05)

    # -- the IBM path is gated: no locks -> raise, submit ZERO jobs ---------
    def test_ibm_path_is_gated(self):
        # a mock service that fails loudly if anything is submitted
        class _NoSubmitService:
            def least_busy(self, *a, **k):
                raise AssertionError("least_busy called: budget would be spent!")

        args = types.SimpleNamespace(ibm=False, ibm_confirm=False,
                                     max_shots=1024, max_circuits=4,
                                     max_total_shots=4096)
        # no locks at all
        with self.assertRaises(ibmp.IBMDisabled):
            ibmp.run_ibm([QuantumCircuit(1)], shots=100, args=args,
                         service=_NoSubmitService())
        self.assertFalse(ibmp.ibm_enabled(args))
        # env set but flags missing -> still disabled
        import os
        prev = os.environ.get("CQFA_IBM_RUN")
        os.environ["CQFA_IBM_RUN"] = "1"
        try:
            with self.assertRaises(ibmp.IBMDisabled):
                ibmp.run_ibm([QuantumCircuit(1)], shots=100, args=args,
                             service=_NoSubmitService())
            # --ibm but not --ibm-confirm -> disabled
            args.ibm = True
            with self.assertRaises(ibmp.IBMDisabled):
                ibmp.run_ibm([QuantumCircuit(1)], shots=100, args=args,
                             service=_NoSubmitService())
        finally:
            if prev is None:
                os.environ.pop("CQFA_IBM_RUN", None)
            else:
                os.environ["CQFA_IBM_RUN"] = prev

    # -- budget cap refuses to over-submit even when fully unlocked ---------
    def test_ibm_budget_cap_refuses(self):
        import os
        class _NoSubmitService:
            def least_busy(self, *a, **k):
                raise AssertionError("least_busy called before cap check!")
        args = types.SimpleNamespace(ibm=True, ibm_confirm=True, max_shots=1024,
                                     max_circuits=4, max_total_shots=100)
        prev = os.environ.get("CQFA_IBM_RUN")
        os.environ["CQFA_IBM_RUN"] = "1"
        try:
            # 2 circuits * 1024 shots = 2048 > max_total_shots=100 -> AssertionError
            with self.assertRaises(AssertionError):
                ibmp.run_ibm([QuantumCircuit(1), QuantumCircuit(1)], shots=1024,
                             args=args, service=_NoSubmitService())
        finally:
            if prev is None:
                os.environ.pop("CQFA_IBM_RUN", None)
            else:
                os.environ["CQFA_IBM_RUN"] = prev


if __name__ == "__main__":
    unittest.main()
