"""
experiments.multitask_benchmark.matrix
============================
Declarative benchmark MATRIX: the (cell x reps x task x models) grid the sweep
executes, plus the rep-key resolver.

Each CELL is (cell_id, rep_keys, task_name, model_names, advantage_claim,
dataset_id, extra). The driver expands it into one master-CSV row per
(cell, rep, model), aggregated over seeds.

Groups (all verified: closed, unitary, power-of-two rep dim):
    z2_rc      Z2 (X(x)Z)         dim 4   |G|=2
    z2_pauli2  Z2 (X(x)X)         dim 4   |G|=2
    z2_pauli3  Z2 (X^(x)3)        dim 8   |G|=2
    z2_pauli4  Z2 (X^(x)4)        dim 16  |G|=2
    c4         C4 shift           dim 4   |G|=4
    c8         C8 shift           dim 8   |G|=8
    d4         D4 (non-abelian)   dim 4   |G|=8

S3 (dim 3, non-power-of-two) is EXCLUDED from the trainable sweep and only
appears in a theory-side coherent-surplus assertion in the tests.
"""
from __future__ import annotations

from cqfa.group import get_representation

ALL4 = ["cqfa", "classical_fa", "non_equivariant", "augmented"]
SWEEP_REPS = ["z2_rc", "z2_pauli2", "z2_pauli3", "z2_pauli4", "c4", "c8", "d4"]
SPT_REPS = ["z2_pauli2", "z2_pauli3", "z2_pauli4"]

# (cell_id, rep_keys, task_name, model_names, advantage_claim, dataset_id, extra)
#
# advantage_claim scope ledger (see README for the full table):
#   B1 expressivity   -- coherent-teacher: cqfa genuinely > classical_fa on the
#                        SMALL Z2 orbits (gap +0.17..+0.19 at the real config);
#                        the gap vanishes for larger orbits (C4/D4), reported
#                        truthfully in the CSV, never hidden.
#   B2 mechanism_only -- surplus-labelled diagnostic.  FINDING: once the
#                        classical baseline is given a TRAINABLE ansatz it stays
#                        competitive on the surplus label (the clean
#                        "classical-FA at chance" separation of cqfa_advantage.py
#                        is a FIXED-observable result and does NOT transfer to a
#                        trainable head). So this cell is labelled mechanism_only,
#                        NOT expressivity -- it documents where the advantage does
#                        not hold, which is part of a complete ledger.
#   B3 inductive_bias -- frame-asymmetry order parameter; equivariant models can
#                        exploit it but augmentation is a competitive control.
#   B4 inductive_bias -- SPT-derived states; NOT the sample-complexity claim.
#   B5 inductive_bias -- genomics RC: exact equivariance + label efficiency;
#                        classical RCPS baseline is competitive (no quantum-advantage).
CELLS = [
    ("B1", SWEEP_REPS, "coherent_teacher",   ALL4, "expressivity",   "synthetic",   {}),
    ("B2", SWEEP_REPS, "orbit_interference", ALL4, "mechanism_only", "synthetic",   {}),
    ("B3", SWEEP_REPS, "symmetry_vs_broken", ALL4, "inductive_bias", "synthetic",   {}),
    ("B4", SPT_REPS,   "spt_derived",        ALL4, "inductive_bias", "spt_states",  {}),
    ("B5", ["z2_rc"],  "genomics_rc",
            ALL4 + ["rc_cnn_classical"], "inductive_bias", "genomics_rc",
            {"k": 2, "seq_len": 12, "motif": "CG", "rho": 0.1}),
]


def resolve_rep(key: str):
    """Map a short matrix rep-key to a built Representation.

    Extends get_representation with the short aliases used in the matrix:
    z2_pauli2/3/4 -> z2_pauli(n_qubits), c8 -> cyclic(8). genomics_rc's U_RC rep
    is built by datasets.genomics_reverse_complement, not here.
    """
    k = key.lower()
    if k == "z2_pauli2":
        return get_representation("z2_pauli", n_qubits=2)
    if k == "z2_pauli3":
        return get_representation("z2_pauli", n_qubits=3)
    if k == "z2_pauli4":
        return get_representation("z2_pauli", n_qubits=4)
    if k == "c8":
        return get_representation("cyclic", n=8)
    return get_representation(key)      # z2_rc, c4, d4 handled by the registry
