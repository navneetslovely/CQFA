
import unittest

import numpy as np

from cqfa import group as G
from cqfa import dynamical_lie_algebra as D


class TestLieClosure(unittest.TestCase):
    def test_su2_closure_rotation_only(self):
        # Rotation generators ALONE close to 3n (commuting single-qubit su(2)s),
        # NOT su(2^n).  Documents the trap (spec Sec 0 / V1).
        for n in (1, 2, 3):
            gens = D.hea_circuit_generators(n, include_entangler=False)
            dim, _ = D.lie_closure_dim(gens)
            self.assertEqual(dim, 3 * n, f"rotation-only DLA n={n}")

    def test_su_full_closure_with_entangler(self):
        # Rotation + CNOT-ring conjugates close to full su(2^n) = 4^n - 1.
        # primary: if this fails the barren-plateau mechanism is wrong.
        for n, expected in ((2, 15), (3, 63), (4, 255)):
            gens = D.hea_circuit_generators(n, include_entangler=True)
            dim, _ = D.lie_closure_dim(gens)
            self.assertEqual(dim, expected, f"circuit DLA n={n}")

    def test_dla_cap_flag(self):
        # Memory guard: the closure never exceeds the cap.
        gens = D.hea_circuit_generators(3, include_entangler=True)
        dim, basis = D.lie_closure_dim(gens, cap=10)
        self.assertEqual(dim, 10)
        self.assertLessEqual(basis.shape[0], 10)


class TestRestrictedDLA(unittest.TestCase):
    def test_frame_induced_dla_scaling(self):
        # z2_pauli(n): restricted DLA g_G is 3(n-1), dim(H_+) is 2^{n-1}.
        for n in (2, 3, 4, 5):
            rep = G.z2_pauli(n)
            d_eff, basis, Bplus = D.frame_induced_dla(rep, n)
            self.assertEqual(d_eff, 3 * (n - 1), f"dim(g_G) n={n}")
            self.assertEqual(Bplus.shape[1], 2 ** (n - 1), f"dim(H+) n={n}")
            self.assertEqual(basis.shape[0], d_eff)

    def test_rc_eqc_dla_is_small(self):
        # Proof anchor (spec Sec 7): z2_rc has dim(H_+)=2, dim(g_G) <= 3.
        rep = G.z2_rc()
        n = int(round(np.log2(rep.dim)))
        d_eff, _, Bplus = D.frame_induced_dla(rep, n)
        self.assertEqual(Bplus.shape[1], 2)
        self.assertLessEqual(d_eff, 3)

    def test_invariant_subspace_is_projector_range(self):
        for rep in (G.z2_rc(), G.z2_pauli(3), G.cyclic(4)):
            Pi = rep.projector()
            B = D.invariant_subspace_basis(rep)
            # Pi B = B (columns lie in range Pi) and B^dag B = I (orthonormal)
            self.assertLess(np.linalg.norm(Pi @ B - B), 1e-9, rep.name)
            self.assertLess(np.linalg.norm(B.conj().T @ B - np.eye(B.shape[1])),
                            1e-9, rep.name)

    def test_symmetrized_generators_commute_with_rep(self):
        rep = G.z2_pauli(3)
        gens = D.symmetrized_generators(D.hea_circuit_generators(3), rep)
        for g in gens:
            for U in rep.matrices:
                self.assertLess(np.linalg.norm(g @ U - U @ g), 1e-9)


class TestGPurity(unittest.TestCase):
    def test_g_purity_of_basis_element(self):
        # For H in g (a basis element), P_g(H) ~ ||H||_HS^2 = 1 (orthonormal).
        gens = D.hea_circuit_generators(2)
        _, basis = D.lie_closure_dim(gens)
        H = basis[0].reshape(4, 4)          # an element of the algebra
        self.assertAlmostEqual(D.g_purity(H, basis), 1.0, places=6)

    def test_g_purity_orthogonal_is_zero(self):
        # Identity is traceless-projected to 0, so g-purity ~ 0.
        gens = D.hea_circuit_generators(2)
        _, basis = D.lie_closure_dim(gens)
        self.assertLess(D.g_purity(np.eye(4, dtype=complex), basis), 1e-9)

    def test_ragone_variance_positive_for_reachable(self):
        rep = G.z2_pauli(3)
        d_eff, basis, Bplus = D.frame_induced_dla(rep, 3)
        # observable in the algebra -> positive prediction with a matching state
        H = basis[0].reshape(Bplus.shape[1], Bplus.shape[1])
        M = -1j * H
        M = (M + M.conj().T) / 2.0
        chi = np.zeros(Bplus.shape[1], dtype=complex)
        chi[0] = 1.0
        rho = np.outer(chi, chi.conj())
        v = D.ragone_variance(rho, M, basis, d_eff)
        self.assertGreaterEqual(v, 0.0)


if __name__ == "__main__":
    unittest.main()
