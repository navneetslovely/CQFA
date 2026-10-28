
import unittest

import numpy as np

from cqfa.group import cyclic
from experiments.discrete_log.dlog_concept import (
    DlogConcept,
    is_prime,
    is_primitive_root,
    smallest_primitive_root,
    discrete_log_table,
    make_dataset,
)
from experiments.discrete_log.learners import (
    fiducial_state,
    covariant_states,
    covariant_kernel,
    structured_accuracy,
    classical_raw_accuracy,
    chance_accuracy,
)
from experiments.discrete_log.run_sweep import run


PRIMES = (7, 11, 23, 47)


class TestNumberTheory(unittest.TestCase):
    def test_is_prime(self):
        for p in PRIMES:
            self.assertTrue(is_prime(p))
        for n in (1, 4, 9, 15, 21, 49, 51):
            self.assertFalse(is_prime(n))

    def test_primitive_root_generates_group(self):
        for p in PRIMES:
            g = smallest_primitive_root(p)
            self.assertTrue(is_primitive_root(g, p))
            # powers g^0..g^{p-2} enumerate Z_p^* exactly once
            powers = {pow(g, e, p) for e in range(p - 1)}
            self.assertEqual(powers, set(range(1, p)), f"p={p}")

    def test_discrete_log_inverts_exponentiation(self):
        for p in PRIMES:
            g = smallest_primitive_root(p)
            logs = discrete_log_table(g, p)
            self.assertEqual(set(logs.keys()), set(range(1, p)), f"p={p}")
            for x in range(1, p):
                self.assertEqual(pow(g, logs[x], p), x, f"p={p} x={x}")


class TestConcept(unittest.TestCase):
    def test_labels_balanced(self):
        # half-arc has exactly (p-1)/2 of the p-1 log values, for every secret s
        for p in PRIMES:
            c = DlogConcept.build(p)
            for s in range(c.order):
                ys = c.labels(c.domain(), s)
                self.assertEqual(int(np.sum(ys == 1)), (p - 1) // 2, f"p={p} s={s}")

    def test_labels_are_one_contiguous_log_arc(self):
        # ordering data by log coordinate, the +1/-1 labels form exactly one arc
        # (2 cyclic sign changes) -- perfectly structured in log, scrambled in x
        for p in PRIMES:
            c = DlogConcept.build(p)
            order_by_log = sorted(c.domain(), key=lambda x: c.logs[x])
            for s in (0, 1, c.order // 2):
                seq = np.array([c.label(x, s) for x in order_by_log])
                sign_changes = int(np.sum(seq != np.roll(seq, 1)))
                self.assertEqual(sign_changes, 2, f"p={p} s={s}")

    def test_label_shift_is_a_log_rotation(self):
        # concept is covariant under log-shift: f_{s+1}(x) == f_s(x') where
        # log(x') = log(x) - 1  (since f_{s+1} tests log(x)-s-1 = f_s at log(x)-1).
        c = DlogConcept.build(23)
        inv_log = {e: x for x, e in c.logs.items()}
        for x in c.domain():
            e = c.logs[x]
            x_shift = inv_log[(e - 1) % c.order]
            self.assertEqual(c.label(x, 1), c.label(x_shift, 0), f"x={x}")

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            DlogConcept.build(15)          # not prime
        with self.assertRaises(ValueError):
            DlogConcept.build(2)           # even prime excluded
        with self.assertRaises(ValueError):
            DlogConcept.build(7, g=6)      # 6 is not a primitive root mod 7


class TestDataset(unittest.TestCase):
    def test_train_test_disjoint_and_labeled(self):
        c = DlogConcept.build(23)
        rng = np.random.default_rng(0)
        x_tr, y_tr, x_te, y_te = make_dataset(c, s=3, n_train=8, n_test=10, rng=rng)
        self.assertEqual(len(set(x_tr.tolist()) & set(x_te.tolist())), 0)
        self.assertTrue(set(np.unique(y_tr).tolist()) <= {-1, 1})
        # labels match the concept exactly
        np.testing.assert_array_equal(y_tr, c.labels(x_tr, 3))
        np.testing.assert_array_equal(y_te, c.labels(x_te, 3))

    def test_oversized_request_raises(self):
        c = DlogConcept.build(7)  # |Z_p^*| = 6
        with self.assertRaises(ValueError):
            make_dataset(c, s=0, n_train=4, n_test=4, rng=np.random.default_rng(0))


class TestCovariantKernel(unittest.TestCase):
    def test_roll_equals_cyclic_rep_matrix_power(self):
        # covariant_states uses np.roll(phi0, k); assert it equals R^k phi0 where
        # R is the *verified* cyclic(order) generator matrix (index 1).
        order = 22
        c = DlogConcept.build(23)
        phi0 = fiducial_state(order)
        R = cyclic(order).matrices[1]
        # find an x whose log is a chosen k, verify state construction matches
        inv_log = {e: x for x, e in c.logs.items()}
        for k in (0, 1, 5, order - 1):
            x = inv_log[k]
            state = covariant_states(c, np.array([x]), phi0)[0]
            expected = np.linalg.matrix_power(R, k) @ phi0
            np.testing.assert_allclose(state, expected, atol=1e-10, err_msg=f"k={k}")

    def test_kernel_depends_only_on_log_difference(self):
        # covariance: K(x,x') is a function of (log x - log x') mod (p-1) only.
        c = DlogConcept.build(23)
        phi0 = fiducial_state(c.order)
        inv_log = {e: x for x, e in c.logs.items()}
        # build K over all pairs; group by log difference; each group is constant
        xs = np.array([inv_log[e] for e in range(c.order)])
        states = covariant_states(c, xs, phi0)
        K = covariant_kernel(states, states)
        by_diff = {}
        for i in range(c.order):
            for j in range(c.order):
                d = (i - j) % c.order
                by_diff.setdefault(d, []).append(K[i, j])
        for d, vals in by_diff.items():
            self.assertLess(np.std(vals), 1e-9, f"diff={d} not constant")

    def test_kernel_psd_and_diag_one(self):
        c = DlogConcept.build(11)
        phi0 = fiducial_state(c.order)
        states = covariant_states(c, c.domain(), phi0)
        K = covariant_kernel(states, states)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)  # fidelity self=1
        eig = np.linalg.eigvalsh((K + K.T) / 2)
        self.assertGreater(eig.min(), -1e-8)  # PSD Gram matrix


class TestSeparation(unittest.TestCase):
    def test_structured_beats_classical_and_chance(self):
        # the primary qualitative result: log-aware kernel generalises,
        # raw-x classical does not (stays near chance). Averaged over seeds.
        # p=47 is the clean regime (classical control sits AT chance).
        c = DlogConcept.build(47)
        phi0 = fiducial_state(c.order)
        s_accs, c_accs, ch_accs = [], [], []
        for seed in range(8):
            rng = np.random.default_rng(seed)
            s = int(rng.integers(0, c.order))
            x_tr, y_tr, x_te, y_te = make_dataset(c, s, n_train=12, n_test=20, rng=rng)
            s_accs.append(structured_accuracy(c, x_tr, y_tr, x_te, y_te, phi0=phi0))
            c_accs.append(classical_raw_accuracy(c, x_tr, y_tr, x_te, y_te))
            ch_accs.append(chance_accuracy(y_te))
        struct = float(np.mean(s_accs))
        classical = float(np.mean(c_accs))
        chance = float(np.mean(ch_accs))
        self.assertGreater(struct, 0.78, f"structured mean acc {struct:.3f} too low")
        # structured clearly separates from the raw-x classical control...
        self.assertGreater(struct - classical, 0.15,
                           f"separation {struct - classical:.3f} too small")
        # ...and the classical control does NOT generalise (near chance).
        self.assertLess(classical - chance, 0.10,
                        f"classical {classical:.3f} implausibly above chance {chance:.3f}")

    def test_run_driver_returns_summary_and_separation(self):
        summary, folder = run(p=47, seeds=8, n_train=12, n_test=20,
                              write=False, verbose=False)
        self.assertEqual(summary["p"], 47)
        self.assertEqual(summary["seeds"], 8)
        self.assertGreater(summary["structured_mean"], 0.78)
        self.assertGreater(summary["separation_mean"], 0.15)
        self.assertIsNone(folder)  # write=False


if __name__ == "__main__":
    unittest.main(verbosity=2)
