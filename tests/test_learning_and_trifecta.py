
import unittest

import numpy as np

from experiments.two_copy_separation import learning_separation as LS
from experiments.trainability_law import trainability_law as T
from experiments.exploratory.trifecta import trifecta as TF


class TestLearningSeparation(unittest.TestCase):
    def test_coherent_samples_are_unbiased_for_purity(self):
        rng = np.random.default_rng(0)
        for pur in (1.0, 0.5, 0.25):
            s = LS.coherent_purity_samples(pur, 20000, rng)
            self.assertAlmostEqual(float(np.mean(s)), pur, delta=0.03)

    def test_pure_and_mixed_purities(self):
        rng = np.random.default_rng(1)
        self.assertAlmostEqual(LS.purity(LS.haar_pure(3, rng)), 1.0, places=6)
        self.assertAlmostEqual(LS.purity(LS.mixed_state(3, rng)), 1 / 8, places=6)  # 2^-3

    def test_shadow_estimate_unbiased(self):
        # single-copy shadow purity estimate is unbiased (just high-variance)
        rng = np.random.default_rng(2)
        rho = LS.haar_pure(2, rng)
        ests = [LS.shadow_purity_estimate(rho, 2, 60, rng) for _ in range(40)]
        self.assertAlmostEqual(float(np.mean(ests)), LS.purity(rho), delta=0.15)

    def test_separation_direction(self):
        # coherent reaches target with far fewer copies than single-copy shadows
        rng = np.random.default_rng(3)
        data = LS.make_dataset(3, 16, rng)
        c_coh, _ = LS.copies_to_accuracy(data, LS.coherent_learner,
                                         [2, 4, 8, 16, 32], rng)
        c_sc, _ = LS.copies_to_accuracy(data, LS.shadow_learner,
                                        [4, 8, 16, 32, 64, 128, 256, 512], rng)
        self.assertIsNotNone(c_coh)
        self.assertTrue(c_sc is None or c_sc > c_coh)


class TestFairAdvantageSuite(unittest.TestCase):
    """The suite of unconditional 2-copy vs single-copy separations (item a)."""

    def test_entanglement_separation_direction(self):
        from experiments.two_copy_separation import tasks
        rng = np.random.default_rng(0)
        data = tasks.make_entanglement_dataset(2, 2, 16, rng)
        c, _ = LS.copies_to_accuracy(data, LS.coherent_learner, [2, 4, 8, 16, 32], rng)
        s, _ = LS.copies_to_accuracy(data, LS.shadow_learner,
                                     [4, 8, 16, 32, 64, 128, 256, 512], rng)
        self.assertIsNotNone(c)
        self.assertTrue(s is None or s > c)

    def test_overlap_dataset_labels(self):
        from experiments.two_copy_separation import tasks
        rng = np.random.default_rng(1)
        data = tasks.make_overlap_dataset(2, 20, rng)
        # equal pairs have overlap 1; independent pairs have small overlap
        eq = [ov for (_, ov, y) in data if y == 1]
        ind = [ov for (_, ov, y) in data if y == 0]
        self.assertTrue(all(abs(o - 1.0) < 1e-9 for o in eq))
        self.assertLess(np.mean(ind), 0.6)


class TestNoiseRobustness(unittest.TestCase):
    """The full noise-channel library + the all-channel robustness of the advantages."""

    def test_all_channels_are_cptp(self):
        from experiments.noise_robustness import noise_lib as NL
        for name, mk in NL.NOISE_MODELS.items():
            for s in (0.0, 0.05, 0.2, 0.5):
                self.assertTrue(NL.is_trace_preserving(mk(s)),
                                f"{name} at s={s} is not CPTP")

    def test_channels_act_correctly(self):
        from experiments.noise_robustness import noise_lib as NL
        # bit_flip(1) on |0><0| -> |1><1|
        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        out = NL.apply_iid(rho0, 1, NL.bit_flip(1.0))
        self.assertAlmostEqual(out[1, 1].real, 1.0, places=6)
        # amplitude_damping(1) sends any state to |0>
        rho1 = np.array([[0, 0], [0, 1]], dtype=complex)
        out = NL.apply_iid(rho1, 1, NL.amplitude_damping(1.0))
        self.assertAlmostEqual(out[0, 0].real, 1.0, places=6)
        # depolarizing(1) -> maximally mixed
        out = NL.apply_iid(rho0, 1, NL.depolarizing(1.0))
        self.assertAlmostEqual(out[0, 0].real, 0.5, places=6)

    def test_twocopy_separation_survives_every_channel(self):
        # the 2-copy purity separation persists (coherent < single-copy) under EACH
        # of the 10 channels at a moderate rate.
        from experiments.noise_robustness.robustness_sweep import _twocopy_unit
        from experiments.noise_robustness import noise_lib as NL
        for model in NL.NOISE_MODELS:
            r = _twocopy_unit(model, 0.05, n=3, m=16, seeds=[0, 1])
            self.assertTrue(np.isnan(r["singlecopy_copies"]) or
                            r["coherent_copies"] < r["singlecopy_copies"],
                            f"{model}: coherent {r['coherent_copies']} !< single {r['singlecopy_copies']}")


class TestSurplusDimension(unittest.TestCase):
    """Surplus-dimension prop with the explicit linear-independence hypothesis."""

    def test_formula_holds(self):
        from cqfa import group as G
        for rep in (G.z2_rc(), G.cyclic(4), G.symmetric(3)):
            d, m = rep.dim, rep.order
            rng = np.random.default_rng(0)
            Mm = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
            M = (Mm + Mm.conj().T) / 2
            allops = np.array([(rep.matrices[g].conj().T @ M @ rep.matrices[h]).reshape(-1)
                               for g in range(m) for h in range(m)])
            diag = np.array([(rep.matrices[g].conj().T @ M @ rep.matrices[g]).reshape(-1)
                             for g in range(m)])
            surplus = np.linalg.matrix_rank(allops, tol=1e-9) - np.linalg.matrix_rank(diag, tol=1e-9)
            self.assertEqual(surplus, min(m * m, d * d) - m, f"group order {m}, dim {d}")


class TestModeF(unittest.TestCase):
    """Mode F (frame Fourier readout): the new component that escapes the trade-off."""

    def test_modes_A_B_blind_to_shift(self):
        from experiments.fisher_information import fourier_mode as F
        N = 8
        rng = np.random.default_rng(0)
        phi = F._rand_state(N, rng)
        M = np.diag(rng.standard_normal(N).astype(complex))
        # Mode A and Mode B outputs are constant across shifts s (blind)
        fa = [F.mode_A(F.shift(N, s) @ phi, N, M)[0] for s in range(N)]
        fb = [F.mode_B(F.shift(N, s) @ phi, N, M) for s in range(N)]
        self.assertLess(np.var(fa), 1e-20)
        self.assertLess(np.var(fb), 1e-20)

    def test_modeF_phase_encodes_shift(self):
        from experiments.fisher_information import fourier_mode as F
        N = 8
        rng = np.random.default_rng(1)
        phi = F._rand_state(N, rng)
        ref, _ = F.mode_F_components(phi, N)
        # k=1 Fourier component phase = -2*pi*s/N
        for s in range(N):
            comps, _ = F.mode_F_components(F.shift(N, s) @ phi, N)
            phase = np.angle(np.vdot(ref[1], comps[1]))
            self.assertAlmostEqual(phase % (2 * np.pi), (-2 * np.pi * s / N) % (2 * np.pi), places=6)

    def test_modeF_efficiency_no_postselection(self):
        from experiments.fisher_information import fourier_mode as F
        rng = np.random.default_rng(2)
        for N in (8, 16, 32):
            psi = F._rand_state(N, rng)
            _, probs = F.mode_F_components(psi, N)
            self.assertAlmostEqual(probs.sum(), 1.0, places=9)   # every shot used

    def test_modeF_learns_shift_A_B_do_not(self):
        from experiments.fisher_information import fourier_mode as F
        rows = F.run([8], m=40, seeds=[0, 1], out_dir="/tmp/_e12test")
        r = rows[0]
        self.assertGreater(r["acc_modeF"], 0.85)
        self.assertLess(r["acc_modeA"], 0.7)
        self.assertLess(r["acc_modeB"], 0.7)

    def test_nonabelian_modeF_reads_sign_irrep(self):
        from experiments.fisher_information import extensions as X
        rows = X.run_nonabelian_s3(m=40, seeds=[0, 1, 2], out_dir="/tmp/_e12test")
        r = rows[0]
        self.assertGreater(r["acc_modeF"], 0.85)          # F reads parity (sign irrep)
        self.assertLess(r["acc_modeA"], 0.7)              # A/B blind to parity
        self.assertLess(r["acc_modeB"], 0.7)

    def test_dlp_trifecta_modeF_learns_trainable(self):
        from experiments.fisher_information import extensions as X
        rows = X.run_dlp_trifecta([11], m=40, seeds=[0, 1], out_dir="/tmp/_e12test")
        r = rows[0]
        self.assertGreater(r["acc_modeF"], 0.85)          # learns hidden-shift label
        self.assertLess(r["acc_modeA"], 0.7)             # projection blind
        self.assertGreater(r["head_grad_var"], 1e-4)     # non-vanishing gradient (trainable)

    def test_real_dlp_quantum_beats_strong_classical(self):
        # REAL discrete-log concept: quantum (dlog) feature separates it; strong
        # tuned classical baselines on raw x fail head-to-head (p>=47 -> chance).
        from experiments.discrete_log import dlp_real as D
        rows = D.run([47], seeds=[0, 1, 2], out_dir="/tmp/_e12test")
        r = rows[0]
        self.assertGreater(r["quantum_dlog_acc"], 0.9)
        self.assertLess(r["best_classical_acc"], 0.65)   # strong classical fails

    def test_circuit_level_dlog_recovers_on_fermat_prime(self):
        # The dlog feature is computed by an ACTUAL Shor circuit (matrix-free exact
        # statevector), not a table: exact recovery on a Fermat prime (r=2^k).
        from experiments.discrete_log import dlp_circuit as C
        s, s_est, ok, nq = C.run_circuit_instance(17, t=6, seed=0)   # 18 qubits
        self.assertEqual(s_est, s)
        self.assertGreaterEqual(nq, 16)


class TestTrifecta(unittest.TestCase):
    def test_dim_hplus_is_constant_four(self):
        for n in (4, 5, 6):
            rep = T.z2k_group(n, n - 2)
            pack = T.dla_pack(rep, n)
            self.assertEqual(pack["dim_hplus"], 4, f"n={n} dim H+ should be 4")

    def test_variance_flat_vs_generic_collapsing(self):
        # analytical Ragone variance is n-independent while generic ~1/4^n
        v4, _ = T.ragone_var_cqfa(T.dla_pack(T.z2k_group(4, 2), 4), np.random.default_rng(0))
        v6, _ = T.ragone_var_cqfa(T.dla_pack(T.z2k_group(6, 4), 6), np.random.default_rng(0))
        self.assertAlmostEqual(v4, v6, places=6)                 # flat in n
        self.assertGreater(v6, T.ragone_var_generic(6) * 10)     # >> generic

    def test_trains_above_chance(self):
        rep = T.z2k_group(5, 3)
        pack = T.dla_pack(rep, 5)
        gens = T.equivariant_ansatz_generators(rep, 5, pack)
        tr, te = TF.train_equivariant(pack, gens, n_train=80, n_test=80,
                                      epochs=60, seed=0)
        self.assertGreater(te, 0.75, "equivariant ansatz should train above chance")


if __name__ == "__main__":
    unittest.main()
