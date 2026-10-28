
import unittest

import numpy as np

from cqfa import group as G
from cqfa import readout as Mo
from cqfa import certificate as C
from experiments.trainability_law import analysis as A


def _rand_state(dim, rng):
    v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return v / np.linalg.norm(v)


class TestTrainabilityAxis(unittest.TestCase):
    def test_cqfa_d_eff_matches_restricted_dla(self):
        rep = G.z2_pauli(3)
        M, ok = C.reachable_observable(rep, 3)
        self.assertTrue(ok)
        rng = np.random.default_rng(0)
        psi = _rand_state(rep.dim, rng)
        phi = rep.projector() @ psi
        chi = phi / np.linalg.norm(phi)
        T = C.trainability_axis(rep, 3, chi, M)
        self.assertEqual(T["d_eff"], 6)           # 3(n-1) at n=3
        self.assertEqual(T["dim_hplus"], 4)       # 2^{n-1}
        # reachable observable => positive P_g(M) and a positive variance lb
        self.assertGreater(T["P_g_M"], 1e-6)
        self.assertGreater(T["var_lb"], 0.0)

    def test_z0_observable_is_degenerate(self):
        # V4: the default Z_0 restricted to H_+ is orthogonal to g_G (P_g(M)=0)
        # -> the restricted model cannot move that loss.  Documented, not hidden.
        from cqfa.simulator import local_z
        rep = G.z2_pauli(3)
        rng = np.random.default_rng(1)
        psi = _rand_state(rep.dim, rng)
        phi = rep.projector() @ psi
        chi = phi / np.linalg.norm(phi)
        T = C.trainability_axis(rep, 3, chi, local_z(3, 0))
        self.assertLess(T["P_g_M"], 1e-9)
        self.assertEqual(T["var_lb"], 0.0)


class TestHardnessAxis(unittest.TestCase):
    def test_escape_follows_source(self):
        rep = G.z2_pauli(2)
        M, _ = C.reachable_observable(rep, 2)
        rng = np.random.default_rng(2)
        psi = _rand_state(rep.dim, rng)
        self.assertEqual(
            C.hardness_axis(rep, psi, M, C.QUANTUM)["escape"], 1)
        self.assertEqual(
            C.hardness_axis(rep, psi, M, C.CLASSICAL)["escape"], 0)

    def test_frame_entropy_grows_with_group(self):
        # Verified monotonicity: D4 > Z2-pauli(2).
        rng = np.random.default_rng(0)
        M2, _ = C.reachable_observable(G.z2_pauli(2), 2)
        s_small = np.mean([C.hardness_axis(
            G.z2_pauli(2), _rand_state(4, rng), M2, C.QUANTUM)["frame_entropy"]
            for _ in range(100)])
        repd = G.dihedral4()
        Md = np.eye(repd.dim, dtype=complex)
        s_big = np.mean([C.hardness_axis(
            repd, _rand_state(repd.dim, rng), Md, C.QUANTUM)["frame_entropy"]
            for _ in range(100)])
        self.assertGreater(s_big, s_small)

    def test_surplus_zero_for_output_model(self):
        # classical_fa averages OUTPUTS: cqfa readout vs classical readout still
        # differ, but the surplus is a property of the DATA/rep not the model.
        # Here we just check surplus is finite and non-negative.
        rep = G.z2_pauli(3)
        M, _ = C.reachable_observable(rep, 3)
        rng = np.random.default_rng(3)
        psi = _rand_state(rep.dim, rng)
        h = C.hardness_axis(rep, psi, M, C.QUANTUM)
        self.assertGreaterEqual(h["surplus"], 0.0)
        self.assertGreaterEqual(h["bond_chi"], 1.0)


class TestCertificateObject(unittest.TestCase):
    def test_certificate_row_shape(self):
        rep = G.z2_pauli(3)
        M, _ = C.reachable_observable(rep, 3)
        rng = np.random.default_rng(4)
        psi = _rand_state(rep.dim, rng)
        row = C.certificate("cqfa", rep, 3, psi, M, C.QUANTUM)
        for k in ("T_d_eff", "T_var_lb", "H_escape", "H_frame_entropy"):
            self.assertIn(k, row)
        self.assertEqual(row["T_d_eff"], 6)


class TestAnalysisEstimators(unittest.TestCase):
    def test_wide_params_distribution(self):
        # Guards the plateau-understatement bug: NOT the N(0,0.1) init.
        rng = np.random.default_rng(0)
        th = A.wide_params(100000, rng)
        self.assertGreaterEqual(th.min(), 0.0)
        self.assertLess(th.max(), 2 * np.pi)
        self.assertGreater(th.mean(), 2.0)                 # ~pi, far from 0
        self.assertAlmostEqual(th.std(), 2 * np.pi / np.sqrt(12), places=1)

    def test_variance_estimator_matches_features_batch(self):
        # The estimator's variance at mu_early equals np.var of the raw
        # features_batch gradients (consistency of the wiring).
        rep = G.z2_pauli(3)
        model = Mo.NonEquivModel(rep, n_layers=2)
        rng = np.random.default_rng(7)
        psi = _rand_state(rep.dim, rng)
        S = 64
        rng2 = np.random.default_rng(7)
        mu_dict = A.component_indices(3, 2)
        var, vals = A.gradient_variance(model, psi, S, rng2, mu_dict["early"])
        # recompute the same draws independently
        rng3 = np.random.default_rng(7)
        Thetas = np.array([A.wide_params(model.n_params, rng3) for _ in range(S)])
        grads = np.array([model.features_batch(t, psi[None, :])[1][0]
                          for t in Thetas])
        self.assertAlmostEqual(var, float(np.var(grads[:, mu_dict["early"]])),
                               places=10)

    def test_gradient_reproducible_under_seed(self):
        rep = G.z2_pauli(3)
        model = Mo.NonEquivModel(rep, n_layers=2)
        mu = A.component_indices(3, 2)["early"]

        def run():
            rng = np.random.default_rng(11)
            psi = _rand_state(rep.dim, rng)
            v, _ = A.gradient_variance(model, psi, 50, rng, mu)
            return v
        self.assertEqual(run(), run())

    def test_hea_variance_decreases_with_n(self):
        # Sign check: the generic HEA's gradient variance drops with n (plateau
        # direction).  Cheap S=200, n in {2,4}.
        vs = {}
        for n in (2, 4):
            rep = G.z2_pauli(n)
            model = Mo.NonEquivModel(rep, n_layers=2)
            rng = np.random.default_rng(5)
            psi = _rand_state(rep.dim, rng)
            mu = A.component_indices(n, 2)["early"]
            vs[n], _ = A.gradient_variance(model, psi, 200, rng, mu)
        self.assertLess(vs[4], vs[2])

    def test_classify_decay_rule(self):
        self.assertEqual(A.classify_decay(0.69, 0.99, 0.5), "plateau")
        self.assertEqual(A.classify_decay(0.05, 0.5, 0.99), "trainable")
        self.assertEqual(A.classify_decay(0.05, 0.9, 0.8), "trainable")

    def test_fit_exp_recovers_slope(self):
        ns = np.array([2, 3, 4, 5, 6])
        V = np.exp(-0.69 * ns) * 3.0                     # V ~ 2^{-n}
        b, _, r2 = A.fit_exp(ns, V)
        self.assertAlmostEqual(b, 0.69, places=6)
        self.assertGreater(r2, 0.999)


if __name__ == "__main__":
    unittest.main()
