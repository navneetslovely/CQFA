import unittest

from cqfa import train as Tr


class TestTrain(unittest.TestCase):
    def test_end_to_end_runs_and_aggregates(self):
        s, _ = Tr.run_experiment("z2_rc", "cqfa", seeds=[0, 1], n_train=30, n_test=20,
                                 epochs=20, lr=0.2, n_layers=2, write=False)
        self.assertEqual(s["n_seeds"], 2)
        self.assertTrue(0.0 <= s["acc_mean"] <= 1.0)
        self.assertEqual(s["n_params"], 2 * 2 * 2)          # n_layers=2, n_qubits=2

    def test_cqfa_beats_classical_fa_on_coherent_task(self):
        common = dict(seeds=[0, 1, 2], n_train=50, n_test=40, epochs=45, lr=0.2, n_layers=3, write=False)
        cq, _ = Tr.run_experiment("z2_rc", "cqfa", **common)
        fa, _ = Tr.run_experiment("z2_rc", "classical_fa", **common)
        self.assertGreater(cq["acc_mean"], fa["acc_mean"])
