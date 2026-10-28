import unittest

import numpy as np

from cqfa import group as G


class TestGroup(unittest.TestCase):
    def reps(self):
        return [G.z2_rc(), G.cyclic(4), G.symmetric(3), G.dihedral4(), G.z2_pauli(3)]

    def test_unitary_closed_identity(self):
        for rep in self.reps():
            self.assertTrue(rep.is_unitary(), rep.name)
            self.assertTrue(rep.is_closed(), rep.name)
            self.assertTrue(rep.has_identity_first(), rep.name)

    def test_orders(self):
        self.assertEqual(G.z2_rc().order, 2)
        self.assertEqual(G.cyclic(4).order, 4)
        self.assertEqual(G.symmetric(3).order, 6)
        self.assertEqual(G.dihedral4().order, 8)
        self.assertEqual(G.z2_pauli(3).order, 2)

    def test_projector_is_idempotent_hermitian(self):
        for rep in self.reps():
            Pi = rep.projector()
            self.assertTrue(np.allclose(Pi @ Pi, Pi, atol=1e-10), rep.name)
            self.assertTrue(np.allclose(Pi, Pi.conj().T, atol=1e-10), rep.name)

    def test_oracle_is_unitary_block_diagonal(self):
        rep = G.cyclic(4)
        V = rep.controlled_oracle()
        dimF = 2 ** rep.n_frame_qubits
        d = rep.dim
        self.assertEqual(V.shape, (dimF * d, dimF * d))
        self.assertTrue(np.allclose(V.conj().T @ V, np.eye(dimF * d), atol=1e-10))

    def test_frame_superposition_normalised(self):
        for rep in self.reps():
            f = rep.frame_superposition()
            self.assertAlmostEqual(float(np.vdot(f, f).real), 1.0, places=10)
