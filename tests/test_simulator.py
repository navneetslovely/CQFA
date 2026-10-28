import unittest

import numpy as np

from cqfa import simulator as S


class TestSimulator(unittest.TestCase):
    def test_param_shift_matches_finite_diff(self):
        rng = np.random.default_rng(4)
        for n, L in [(2, 2), (3, 3)]:
            ans = S.Ansatz(n, L)
            O = S.local_z(n, 0)
            x = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
            x /= np.linalg.norm(x)
            th = rng.standard_normal(ans.n_params)
            _, g = ans.expval_grad(th, x, O)
            eps = 1e-6
            gfd = np.array([(ans.expval(th + eps * e, x, O) - ans.expval(th - eps * e, x, O)) / (2 * eps)
                            for e in np.eye(len(th))])
            self.assertLess(np.max(np.abs(g - gfd)), 1e-6)

    def test_ansatz_unitary(self):
        ans = S.Ansatz(3, 3)
        th = np.random.default_rng(0).standard_normal(ans.n_params)
        W = ans.unitary(th)
        self.assertLess(np.max(np.abs(W.conj().T @ W - np.eye(8))), 1e-10)

    def test_batch_matches_single(self):
        rng = np.random.default_rng(5)
        n, L = 3, 2
        ans = S.Ansatz(n, L)
        O = S.local_z(n, 0)
        X = rng.standard_normal((10, 2 ** n)) + 1j * rng.standard_normal((10, 2 ** n))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        th = rng.standard_normal(ans.n_params)
        vb, gb = ans.expvals_grad(th, X, O)
        for i in range(10):
            v, g = ans.expval_grad(th, X[i], O)
            self.assertAlmostEqual(v, vb[i], places=9)
            self.assertLess(np.max(np.abs(g - gb[i])), 1e-9)
