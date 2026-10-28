
import unittest

import numpy as np

from cqfa import group as G
from cqfa import readout as Mo


class TestRCEQCRegression(unittest.TestCase):
    def test_projector_is_the_rc_projector(self):
        rep = G.z2_rc()
        A = rep.matrices[1]
        self.assertTrue(np.allclose(A @ A, np.eye(4), atol=1e-12))          # involution
        self.assertTrue(np.allclose(rep.projector(), (np.eye(4) + A) / 2, atol=1e-12))

    def test_exact_rc_invariance_for_all_parameters(self):
        rep = G.z2_rc()
        model = Mo.CQFAModel(rep, n_layers=3)
        A = rep.matrices[1]
        rng = np.random.default_rng(7)
        for _ in range(5):
            th = rng.standard_normal(model.n_params)
            for _ in range(20):
                psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
                psi /= np.linalg.norm(psi)
                f0 = model.feature(th, psi)[0]
                f1 = model.feature(th, A @ psi)[0]
                self.assertAlmostEqual(f0, f1, places=10)
