
import csv
import os
import tempfile
import unittest

import numpy as np

from cqfa.group import get_representation, symmetric
from cqfa import frame as L
from cqfa import tasks as T
from datasets import genomics_reverse_complement as GR
from datasets import rc_symmetric_baseline as RCB
from experiments.multitask_benchmark import run_benchmarks as RUN
from experiments.multitask_benchmark.matrix import SWEEP_REPS, resolve_rep


class TestSweepReps(unittest.TestCase):
    def test_all_reps_power_of_two(self):
        for key in SWEEP_REPS:
            rep = resolve_rep(key)
            n = np.log2(rep.dim)
            self.assertEqual(2 ** int(round(n)), rep.dim, f"{key} not power-of-two")
            self.assertTrue(rep.is_closed(), f"{key} not closed")
            self.assertTrue(rep.is_unitary(), f"{key} not unitary")


class TestTaskContracts(unittest.TestCase):
    def _check(self, X, y, dim):
        self.assertEqual(X.shape[1], dim)
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertTrue(np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-9),
                        "rows must be unit-norm states")
        self.assertTrue(set(np.unique(y)).issubset({0, 1}))
        bal = float(np.mean(y))
        self.assertTrue(0.3 <= bal <= 0.7, f"class balance {bal} out of [0.3,0.7]")

    def test_orbit_interference_contract(self):
        rep = get_representation("z2_rc")
        fn = T.make_task("orbit_interference", rep)
        X, y = fn(60, np.random.default_rng(0))
        self._check(X, y, rep.dim)

    def test_symmetry_vs_broken_contract(self):
        for key in ("z2_rc", "c8", "d4"):        # incl. the high-q concentration reps
            rep = resolve_rep(key)
            fn = T.make_task("symmetry_vs_broken", rep)
            X, y = fn(60, np.random.default_rng(1))
            self._check(X, y, rep.dim)

    def test_spt_derived_contract(self):
        rep = get_representation("z2_pauli", n_qubits=2)
        fn = T.make_task("spt_derived", rep)
        X, y = fn(60, np.random.default_rng(2))
        self._check(X, y, rep.dim)


class TestExpressivitySeparation(unittest.TestCase):
    def test_cqfa_beats_classical_fa_on_z2_coherent_teacher(self):
       
        from cqfa import train as Tr
        common = dict(n_train=64, n_test=48, epochs=80, lr=0.2, n_layers=3, write=False)
        rep = get_representation("z2_rc")
        fn = T.make_task("coherent_teacher", rep)
        cq, _ = Tr.run_experiment("z2_rc", "cqfa", [0, 1, 2], task_fn=fn,
                                  task_name="coherent_teacher", rep=rep, **common)
        fa, _ = Tr.run_experiment("z2_rc", "classical_fa", [0, 1, 2], task_fn=fn,
                                  task_name="coherent_teacher", rep=rep, **common)
        self.assertGreater(cq["acc_mean"], fa["acc_mean"] + 0.03)

    def test_orbit_interference_task_is_valid(self):
    
        from cqfa import train as Tr
        common = dict(n_train=48, n_test=32, epochs=40, lr=0.2, n_layers=3, write=False)
        rep = get_representation("z2_rc")
        fn = T.make_task("orbit_interference", rep)
        cq, _ = Tr.run_experiment("z2_rc", "cqfa", [0, 1], task_fn=fn,
                                  task_name="orbit_interference", rep=rep, **common)
        self.assertTrue(0.0 <= cq["acc_mean"] <= 1.0)


class TestGenomicsRC(unittest.TestCase):
    def test_rc_invariance_and_encoding(self):
        rng = np.random.default_rng(0)
        X, y, rep, meta, seqs = GR.build_genomics_rc(60, np.random.default_rng(3),
                                                     k=2, seq_len=12)
        # U_RC involution
        U = rep.matrices[1]
        self.assertTrue(np.allclose(U @ U, np.eye(rep.dim), atol=1e-12))
        # unit-norm rows
        self.assertTrue(np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-9))
        # label RC-invariance for every row (primary correctness check)
        for seq, lab in zip(seqs, y):
            y_rc = 1 if GR._contains_motif_rc_invariant(
                GR.reverse_complement(seq), meta["motif"]) else 0
            self.assertEqual(y_rc, int(lab))
        # group action == reverse complement on the encoded state
        for seq in seqs[:10]:
            s = GR._window_to_state(seq, 2, rng)
            s_rc = GR._window_to_state(GR.reverse_complement(seq), 2, rng)
            self.assertTrue(np.allclose(U @ s, s_rc, atol=1e-12))

    def test_rc_cnn_baseline_runs(self):
        r = RCB.run_rc_baseline_seed(0, 48, 32, k=2)
        self.assertTrue(0.0 <= r["acc"] <= 1.0)
        self.assertGreater(r["n_params"], 0)


class TestDriver(unittest.TestCase):
    def test_driver_quick_all_cells(self):
        with tempfile.TemporaryDirectory() as td:
            out_csv = os.path.join(td, "benchmark_master.csv")
            runs_root = os.path.join(td, "runs")
            RUN.main(["--quick", "--out", out_csv, "--runs-root", runs_root])
            self.assertTrue(os.path.exists(out_csv))
            with open(out_csv) as fh:
                rows = list(csv.DictReader(fh))
            # 101 model-runs: B1=28, B2=28, B3=28, B4=12, B5=5
            self.assertEqual(len(rows), 101)
            cells = {r["cell"] for r in rows}
            self.assertEqual(cells, {"B1", "B2", "B3", "B4", "B5"})
            # per-cell run folders created for a quantum cell
            self.assertTrue(os.path.isdir(os.path.join(runs_root, "B1")))
            # every quantum row must have status ok and a run folder
            for r in rows:
                if r["quantum_model"] == "True":
                    self.assertEqual(r["status"], "ok")
                    self.assertTrue(r["run_folder"])

    def test_master_csv_schema(self):
        with tempfile.TemporaryDirectory() as td:
            out_csv = os.path.join(td, "m.csv")
            RUN.run([0], cells_filter=["B4"],
                    cfg=dict(n_train=16, n_test=16, epochs=10, lr=0.2, n_layers=2),
                    out_csv=out_csv, runs_root=os.path.join(td, "runs"), verbose=False)
            with open(out_csv) as fh:
                header = next(csv.reader(fh))
            self.assertEqual(header, RUN.MASTER_FIELDS)


class TestS3TheorySurplus(unittest.TestCase):
    def test_s3_coherent_surplus_nonzero(self):

        rep = symmetric(3)
        rng = np.random.default_rng(0)
        M = T.random_hermitian(rep.dim, np.random.default_rng(99))
        surpluses = []
        for _ in range(50):
            psi = T.random_states(rep.dim, 1, rng)[0]
            fa, p = L.cqfa_readout(psi, rep, M)
            if fa is None:
                continue
            fb = L.classical_fa_readout(psi, rep, M)
            surpluses.append(abs(fa - fb))
        self.assertGreater(max(surpluses), 1e-6,
                           "coherent surplus must be nonzero on S3")


if __name__ == "__main__":
    unittest.main()
