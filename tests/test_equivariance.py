import unittest

import numpy as np

from cqfa import group as G
from cqfa import frame as L
from cqfa import verify as V


class TestEquivariance(unittest.TestCase):
    def test_theorem1_exact_invariance(self):
        for rep in [G.z2_rc(), G.cyclic(4), G.symmetric(3), G.dihedral4()]:
            r = V.verify_equivariance(rep, n_inputs=40)
            self.assertLess(r["err_fa"], 1e-9, rep.name)
            self.assertLess(r["err_cqfa"], 1e-9, rep.name)
            self.assertGreater(r["err_base"], 1e-3, rep.name)

    def test_postselection_equals_projector(self):
        rng = np.random.default_rng(2)
        for rep in [G.z2_rc(), G.cyclic(4), G.dihedral4()]:
            for _ in range(20):
                psi = rng.standard_normal(rep.dim) + 1j * rng.standard_normal(rep.dim)
                psi /= np.linalg.norm(psi)
                data, p = L.postselect_symmetric(psi, rep)
                phi = rep.projector() @ psi
                if data is not None:
                    self.assertLess(np.linalg.norm(data - phi), 1e-10, rep.name)

    def test_theorem2_separation(self):
        r = V.verify_separation(G.z2_pauli(3), n_inputs=1000)
        self.assertLess(r["surplus_symmetric"], 1e-9)
        self.assertGreater(r["surplus_generic"], 0.1)
