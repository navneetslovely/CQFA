
import json
import os
import tempfile
import types
import unittest

import numpy as np

from experiments.hardware_ibm.ibm_campaign import specs as S
from experiments.hardware_ibm.ibm_campaign import runner as R
from experiments.hardware_ibm.ibm_campaign import analyze as A
from experiments.hardware_ibm.ibm_campaign import theory as TH
from experiments.hardware_ibm.ibm_campaign import campaign as C
from experiments.hardware_ibm.ibm_runtime_path import IBMDisabled


# small, fast campaign: one SPT size, few shadow snapshots, single shot level
SMALL_OVERRIDES = {
    "C1": {"sizes": (4, 5), "n_shadow": 30, "shots": 4096},
    "C2": {"shots": 8192},
    "C3": {"shots_list": (4096,)},
    "C5": {"shots": 8192, "input_seeds": (5,)},
    "C6": {"shots": 8192},
    "C7": {"shots": 4096, "layers": (0, 1)},
    "C8": {"frame_shifts": ((2, (1, 3)), (3, (2,))), "shots": 4096},
}


def _run_ideal(which=("C1", "C2", "C3", "C4", "C5", "C6", "C7")):
    specs = S.build_campaign(which=which, overrides=SMALL_OVERRIDES)
    res = R.run_offline(specs, backend_kind="aer_noiseless", seed=7)
    analyzed = A.analyze_all(specs, res["counts_by_id"])
    return specs, res, analyzed


class TestCampaignCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs, cls.res, cls.analyzed = _run_ideal()

    def _tol(self, nshots, floor=0.06):
        return max(floor, 6.0 / np.sqrt(max(1, nshots)))

    def test_c1_string_matches_theory(self):
        for r in self.analyzed["C1"]:
            self.assertLessEqual(r["abs_err"], self._tol(r["shots"]),
                                 f"C1 {r['pub_id']} err {r['abs_err']:.3f}")

    def test_c1_string_sign_is_pm1_for_spt(self):
        # cluster state is a +/-1 eigenstate of the measured Y-decorated string;
        # trivial paramagnet gives 0. Sign alternates with n (the fix).
        for r in self.analyzed["C1"]:
            if r["phase"] == "spt":
                self.assertAlmostEqual(abs(r["string_order_exact"]), 1.0, places=6)
            else:
                self.assertAlmostEqual(r["string_order_exact"], 0.0, places=6)

    def test_c1_shadow_hit_rate_tracks_inverse_3w(self):
        row = self.analyzed["C1_shadow"][0]
        self.assertEqual(row["three_pow_w"], 81)          # w=4 at n=4
        self.assertLess(row["hit_rate"], 0.1)             # rare hits ~ 1/81

    def test_c2_purity_matches_theory(self):
        for r in self.analyzed["C2"]:
            self.assertLessEqual(r["abs_err"], self._tol(r["shots"]),
                                 f"C2 {r['pub_id']} err {r['abs_err']:.3f}")

    def test_c2_purity_formula(self):
        self.assertAlmostEqual(TH.c2_purity_theory_mixed(0.0), 1.0, places=9)
        self.assertAlmostEqual(TH.c2_purity_theory_mixed(np.pi / 4), 0.5, places=9)

    def test_c3_postselection_matches_exact(self):
        for r in self.analyzed["C3"]:
            self.assertLessEqual(r["abs_err_postsel"], self._tol(r["n_accepted"], 0.08))

    def test_c4_surplus_large_and_control_zero(self):
        by = {r["input_id"]: r for r in self.analyzed["C4"]}
        self.assertGreater(abs(by["maxsurplus"]["surplus_exact"]), 0.3)
        self.assertLessEqual(abs(by["maxsurplus"]["abs_err_surplus"]), 0.1)
        self.assertLess(abs(by["control"]["surplus_exact"]), 0.05)

    def test_c5_equivariance_gap_small(self):
        for r in self.analyzed["C5"]:
            self.assertLessEqual(r["equivariance_gap_vs_g0"],
                                 self._tol(r["shots"], 0.12))

    def test_c6_multiframe_matches_exact(self):
        self.assertTrue(self.analyzed["C6"])
        for r in self.analyzed["C6"]:
            self.assertLessEqual(r["abs_err_postsel"], self._tol(r["n_accepted"], 0.12))


class TestC8ModeF(unittest.TestCase):
    """C8 (Mode F frame Fourier readout): the ideal sampler recovers the hidden
    shift under Mode F (peak at k=s) and is uniformly blind under Mode B."""

    @classmethod
    def setUpClass(cls):
        cls.specs, cls.res, cls.analyzed = _run_ideal(which=("C8",))

    def test_mode_f_recovers_hidden_shift(self):
        F = [r for r in self.analyzed["C8"] if r["mode"] == "F"]
        self.assertTrue(F)
        for r in F:
            self.assertEqual(r["peak_k"], r["hidden_shift"],
                             f"{r['pub_id']} peak {r['peak_k']} != s {r['hidden_shift']}")
            self.assertEqual(r["success"], 1)
            self.assertGreater(r["peak_prob"], 0.95)      # ideal delta at s

    def test_mode_b_is_blind_uniform(self):
        B = [r for r in self.analyzed["C8"] if r["mode"] == "B"]
        self.assertTrue(B)
        for r in B:
            self.assertLess(r["tv_from_uniform"], 0.1)     # decohered frame -> uniform
            self.assertLess(r["peak_prob"], 2.0 / r["N"] + 0.05)  # no concentration

    def test_theory_from_built_circuit_is_delta_at_s(self):
        # exact per-circuit statevector theory: Mode F ideal peak == s (the gate's
        # ground truth), Mode B ideal is uniform.
        for s in self.specs:
            if s.meta["mode"] == "F":
                self.assertEqual(s.theory["ideal_peak_k"], s.meta["hidden_shift"])
                self.assertGreater(s.theory["ideal_peak_prob"], 0.99)
            else:
                self.assertTrue(s.theory["ideal_uniform"])


class TestCampaignGuards(unittest.TestCase):
    def setUp(self):
        self.specs = S.build_campaign(which=("C3",), overrides=SMALL_OVERRIDES)

    def test_ibm_refuses_without_locks(self):
        args = types.SimpleNamespace(ibm=False, ibm_confirm=False, budget_seconds=300.0)
        with self.assertRaises(IBMDisabled):
            R.run_ibm(self.specs, args)

    def test_ibm_refuses_with_partial_locks(self):
        os.environ["CQFA_IBM_RUN"] = "1"
        try:
            args = types.SimpleNamespace(ibm=True, ibm_confirm=False, budget_seconds=300.0)
            with self.assertRaises(IBMDisabled):
                R.run_ibm(self.specs, args)
        finally:
            os.environ.pop("CQFA_IBM_RUN", None)

    def test_budget_gate(self):
        with self.assertRaises(R.BudgetExceeded):
            R.enforce_budget(self.specs, 0.001)
        cost = R.enforce_budget(self.specs, 10_000.0)
        self.assertGreater(cost["total_shots"], 0)

    def test_token_scrub_blocks_leak(self):
        os.environ["QISKIT_IBM_TOKEN"] = "SECRET_TOKEN_ABC123"
        try:
            with self.assertRaises(RuntimeError):
                C._scrub_token("payload with SECRET_TOKEN_ABC123 inside")
            # normal payload passes
            self.assertIn("ok", C._scrub_token("all ok here"))
        finally:
            os.environ.pop("QISKIT_IBM_TOKEN", None)

    def test_cost_estimate_fields(self):
        cost = R.estimate_cost(self.specs)
        self.assertEqual(cost["n_pubs"], len(self.specs))
        self.assertIn("C3", cost["by_experiment"])
        self.assertAlmostEqual(
            cost["est_qpu_seconds"],
            round(R.JOB_OVERHEAD_SECONDS + R.PER_SHOT_SECONDS * cost["total_shots"], 2),
            places=6)


class TestCalibrationSnapshot(unittest.TestCase):
    """Exercise the 2q-gate calibration extraction against a real Heron Target
    (the device-only branch is otherwise never hit by the offline paths)."""

    def test_snapshot_captures_1q_and_2q_errors(self):
        from qiskit_ibm_runtime.fake_provider import FakeFez
        fake = FakeFez()
        edge = next(iter(fake.coupling_map))          # a genuinely-coupled pair
        a, b = int(edge[0]), int(edge[1])
        snap = R._calibration_snapshot(fake, {a, b}, {(min(a, b), max(a, b))},
                                       source="layout")
        self.assertEqual(snap["used_qubits_source"], "layout")
        self.assertEqual(snap["used_qubits"], sorted({a, b}))
        # single-qubit props present
        self.assertIn(str(a), snap["qubit_properties"])
        # two-qubit gate error for the used edge recorded (ecr/cz/cx)
        self.assertIn("two_qubit_gates", snap)
        key = f"{min(a, b)}_{max(a, b)}"
        self.assertIn(key, snap["two_qubit_gates"])
        self.assertTrue(any("error" in k for k in snap["two_qubit_gates"][key]),
                        f"no 2q gate error captured: {snap['two_qubit_gates'][key]}")


class TestReanalysisRoundTrip(unittest.TestCase):
    def test_saved_counts_reanalyze_matches(self):
        which = ("C1", "C2", "C4")
        specs = S.build_campaign(which=which, overrides=SMALL_OVERRIDES)
        res = R.run_offline(specs, backend_kind="aer_noiseless", seed=3)
        first = A.analyze_all(specs, res["counts_by_id"])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "raw_counts.json")
            with open(path, "w") as f:
                json.dump(C._jsonable(res["counts_by_id"]), f)
            with open(path) as f:
                reloaded = json.load(f)
            specs2 = S.build_campaign(which=which, overrides=SMALL_OVERRIDES)
            second = A.analyze_all(specs2, reloaded)
        # deterministic rebuild -> identical surplus + purity rows
        self.assertEqual(len(first["C2"]), len(second["C2"]))
        b1 = {r["input_id"]: r["surplus_hat"] for r in first["C4"]}
        b2 = {r["input_id"]: r["surplus_hat"] for r in second["C4"]}
        self.assertEqual(b1.keys(), b2.keys())


if __name__ == "__main__":
    unittest.main()
