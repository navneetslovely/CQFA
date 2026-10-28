import unittest

import numpy as np

from cqfa import group as G
from cqfa import readout as Mo


class TestModels(unittest.TestCase):
    def test_matched_parameter_counts(self):
        counts = {k: v.n_params for k, v in Mo.build_models(G.z2_rc(), n_layers=3).items()}
        self.assertEqual(len(set(counts.values())), 1, counts)

    def test_cqfa_and_fa_are_equivariant(self):
        rep = G.z2_rc()
        ms = Mo.build_models(rep, 2)
        rng = np.random.default_rng(1)
        th = rng.standard_normal(ms["cqfa"].n_params)
        for _ in range(10):
            psi = rng.standard_normal(rep.dim) + 1j * rng.standard_normal(rep.dim)
            psi /= np.linalg.norm(psi)
            for U in rep.matrices:
                self.assertAlmostEqual(ms["cqfa"].feature(th, U @ psi)[0],
                                       ms["cqfa"].feature(th, psi)[0], places=9)
                self.assertAlmostEqual(ms["classical_fa"].feature(th, U @ psi)[0],
                                       ms["classical_fa"].feature(th, psi)[0], places=9)

    def test_non_equivariant_is_not_invariant(self):
        rep = G.z2_rc()
        m = Mo.build_models(rep, 2)["non_equivariant"]
        rng = np.random.default_rng(2)
        th = rng.standard_normal(m.n_params)
        A = rep.matrices[1]
        diffs = []
        for _ in range(10):
            psi = rng.standard_normal(rep.dim) + 1j * rng.standard_normal(rep.dim)
            psi /= np.linalg.norm(psi)
            diffs.append(abs(m.feature(th, A @ psi)[0] - m.feature(th, psi)[0]))
        self.assertGreater(max(diffs), 1e-3)

    def test_features_batch_matches_single(self):
        rep = G.z2_rc()
        ms = Mo.build_models(rep, 2)
        rng = np.random.default_rng(3)
        th = rng.standard_normal(ms["cqfa"].n_params)
        X = rng.standard_normal((8, rep.dim)) + 1j * rng.standard_normal((8, rep.dim))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        for name, m in ms.items():
            fb, gb = m.features_batch(th, X)
            for i in range(8):
                f, g = m.feature(th, X[i])
                self.assertAlmostEqual(f, fb[i], places=8, msg=name)
                self.assertLess(np.max(np.abs(g - gb[i])), 1e-8, name)
