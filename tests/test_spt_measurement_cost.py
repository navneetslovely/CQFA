
import unittest

import numpy as np

from experiments.spt_measurement_cost import spt_model as M
from experiments.spt_measurement_cost import readout_coherent as C
from experiments.spt_measurement_cost import readout_shadow as S
from experiments.spt_measurement_cost import string_family as F
from experiments.spt_measurement_cost import sample_complexity as SC


# ---------------------------------------------------------------------------
class TestPhysics(unittest.TestCase):
    def test_z2xz2_commutes(self):
        for n in (6, 8):
            self.assertLess(M.z2xz2_commutator_max(n, 0.7), 1e-9, f"n={n}")

    def test_genuine_transition(self):
        # a real collapse, not a monotone smooth decay
        n = 10
        s_spt = M.string_order(M.ground_state(n, 0.3), n)
        s_mid = M.string_order(M.ground_state(n, 1.0), n)
        s_tri = M.string_order(M.ground_state(n, 2.5), n)
        self.assertGreater(s_spt, 0.9)
        self.assertLess(abs(s_tri), 0.05)
        self.assertLess(s_mid, 0.5)          # collapses through the transition

    def test_string_order_cluster(self):
        for n in (6, 8):
            psi = M.cluster_state(n)
            self.assertAlmostEqual(M.string_order(psi, n), 1.0, places=9, msg=f"n={n}")

    def test_naive_string_is_zero(self):
   
        for n in (6, 8):
            Nop = M.naive_string_operator(n)
            # anticommutation with the protecting symmetry (the physics reason)
            pe, po = M.z2xz2_generators(n)
            self.assertLess(np.max(np.abs(Nop @ pe + pe @ Nop)), 1e-9, f"n={n} even")
            self.assertLess(np.max(np.abs(Nop @ po + po @ Nop)), 1e-9, f"n={n} odd")
            # exactly zero on the cluster state (definite parity)
            psi_c = M.cluster_state(n)
            self.assertLess(abs(float(np.real(psi_c.conj() @ Nop @ psi_c))), 1e-9, f"n={n}")
            # exactly zero on a definite-parity (P_even = +1) trivial ground state.
            # (Project the ED ground state onto the +1 sector so the anticommutation
            #  argument bites exactly; without projection a finite-size degenerate
            #  sector can leak an O(1/exp) artifact -- not a real order parameter.)
            psi_t = M.ground_state(n, 2.5)
            proj = 0.5 * (np.eye(2 ** n) + pe) @ psi_t
            proj = proj / np.linalg.norm(proj)
            self.assertLess(abs(float(np.real(proj.conj() @ Nop @ proj))), 1e-9, f"n={n}")

    def test_local_cannot_separate(self):
        n = 8
        s_spt = M.string_order(M.cluster_state(n), n)
        s_tri = M.string_order(M.ground_state(n, 2.5), n)
        z_spt = M.local_z_mid(M.cluster_state(n), n)
        z_tri = M.local_z_mid(M.ground_state(n, 2.5), n)
        # string order separates strongly; the local <Z_mid> does not
        self.assertGreater(abs(s_spt - s_tri), 0.9)
        self.assertLess(abs(z_spt - z_tri), 0.1)


# ---------------------------------------------------------------------------
class TestReadout(unittest.TestCase):
    def test_two_copy_identity(self):
        rng = np.random.default_rng(0)
        for n in (6, 8):
            psi = M.cluster_state(n)
            O = M.string_operator(n)
            self.assertAlmostEqual(C.two_copy_expectation(psi, O),
                                   M.string_order(psi, n) ** 2, places=9)
            # random Hermitian P^2!=I still satisfies the identity form
            A = rng.standard_normal((2 ** n, 2 ** n)) + 1j * rng.standard_normal((2 ** n, 2 ** n))
            P = (A + A.conj().T) / 2
            e = float(np.real(psi.conj() @ P @ psi))
            self.assertAlmostEqual(C.two_copy_expectation(psi, P), e * e, places=9)

    def test_purity_swaptest(self):
        n = 6
        psi = M.cluster_state(n)
        self.assertAlmostEqual(C.purity(psi), 1.0, places=12)
        self.assertAlmostEqual(C.purity_swaptest(psi), 1.0, places=12)
        # genuine mixture: swap-test identity Tr(rho^2)
        a = M.cluster_state(n)
        b = M.ground_state(n, 2.5)
        w = [0.6, 0.4]
        pur = C.purity_mixture([a, b], w)
        # <psi_a|psi_b>^2 route sanity: purity of the mixture is between the pure
        # value and 0.5 for a 2-component mixture
        self.assertGreater(pur, 0.5 - 1e-9)
        self.assertLessEqual(pur, 1.0 + 1e-9)

    def test_frame_register_swaptest(self):
        # explicit controlled-SWAP-off-frame-qubit circuit agrees with |<a|b>|^2
        rng = np.random.default_rng(1)
        d = 4
        a = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        a /= np.linalg.norm(a)
        b = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        b /= np.linalg.norm(b)
        self.assertAlmostEqual(C.frame_register_swaptest(a, b),
                               C.purity_swaptest(a, b), places=9)

    def test_bell_estimator_unbiased(self):
        n = 6
        psi = M.ground_state(n, 0.3)
        O = M.string_operator(n)
        exact = M.string_order(psi, n) ** 2
        rng = np.random.default_rng(2)
        N = 20000
        samples = C.bell_sample(psi, O, rng, N)
        est = samples.mean()
        std = samples.std(ddof=1)
        self.assertLess(abs(est - exact), 3 * std / np.sqrt(N) + 1e-3)

    def test_no_4n_object(self):
        # the coherent 2-copy path must never allocate a >= 2^(2n) array
        n = 6
        psi = M.cluster_state(n)
        O = M.string_operator(n)
        rng = np.random.default_rng(3)
        with C.FourNTripwire(n):
            _ = C.two_copy_expectation(psi, O)
            _ = C.purity(psi)
            _ = C.bell_estimate(psi, O, rng, 100)
            _ = C.coherent_feature(psi, [O], rng, 100)
        # tripwire actually trips on a real 4^n allocation
        with self.assertRaises(MemoryError):
            with C.FourNTripwire(n):
                np.zeros(2 ** (2 * n), dtype=complex)

    def test_string_label_matches_operator(self):
        for n in (6, 8):
            lab = S.string_pauli_label(n)
            paulis = {"X": M.X, "Y": M.Y, "Z": M.Z}
            Pl = M.op_on({s: paulis[p] for s, p in lab.items()}, n)
            O = M.string_operator(n)
            self.assertTrue(np.allclose(Pl, O) or np.allclose(Pl, -O), f"n={n}")

    def test_shadow_expectation_unbiased(self):
        n = 6
        psi = M.ground_state(n, 0.3)
        rng = np.random.default_rng(4)
        snaps = S.shadow_snapshots(psi, rng, 40000)
        lab = S.string_pauli_label(n)
        exact = M.string_order(psi, n)
        est = S.estimate_expectation(snaps, lab)
        # weight-n string: high variance, but should be within a few sigma
        self.assertLess(abs(est - exact), 0.15)

    def test_direct_known_string_exact_on_cluster(self):
        # cluster state is a +1 eigenstate of O_string -> direct measurement = 1 exactly
        n = 8
        psi = M.cluster_state(n)
        rng = np.random.default_rng(5)
        self.assertAlmostEqual(S.known_string_estimate(psi, n, rng, 500), 1.0, places=9)


# ---------------------------------------------------------------------------
class TestSeparation(unittest.TestCase):
    BUDGETS = [16, 64, 256, 1024, 4096]
    SEEDS = [0, 1]
    EPS = 0.1

    NS = (6, 8)   # CI keeps the gate fast; the smoke/full runs sweep n up to 10/12

    @classmethod
    def setUpClass(cls):
        cls.cstar = {}
        for arm in ("coherent", "shadow_nonlinear", "shadow_known"):
            cls.cstar[arm] = {}
            for n in cls.NS:
                agg = SC.cstar_over_seeds(arm, n, cls.BUDGETS, cls.EPS, cls.SEEDS,
                                          n_boot=200)
                cls.cstar[arm][n] = agg

    def _means(self, arm, ns):
        return [self.cstar[arm][n]["cstar_mean"] for n in ns]

    def test_coherent_flat(self):
        ns = self.NS
        slope = SC.log_cstar_slope(ns, self._means("coherent", ns))
        self.assertLess(abs(slope), 0.5, f"coherent slope {slope} not flat")

    def test_shadow_nonlinear_grows(self):
        ns = self.NS
        means = self._means("shadow_nonlinear", ns)
        # strictly increasing and positive log-slope
        self.assertGreater(means[-1], means[0])
        self.assertGreater(SC.log_cstar_slope(ns, means), 0.0)

    def test_known_string_is_flat(self):
        # INTEGRITY GUARD: the known-string control must not blow up with n
        ns = self.NS
        slope = SC.log_cstar_slope(ns, self._means("shadow_known", ns))
        self.assertLess(abs(slope), 0.5, f"known-string slope {slope} not flat")

    def test_coherent_beats_shadow_at_large_n(self):
        n = self.NS[-1]
        coh = self.cstar["coherent"][n]["cstar_mean"]
        sha = self.cstar["shadow_nonlinear"][n]["cstar_mean"]
        self.assertGreater(sha / coh, 1.0)


# ---------------------------------------------------------------------------
class TestFunctional(unittest.TestCase):
    def test_functional_separates_phases(self):
        for n in (6, 8):
            f_spt = F.functional_normalized_exact(M.cluster_state(n), n)
            f_tri = F.functional_normalized_exact(M.ground_state(n, 2.5), n)
            self.assertGreater(f_spt, 0.9)
            self.assertLess(f_tri, 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
