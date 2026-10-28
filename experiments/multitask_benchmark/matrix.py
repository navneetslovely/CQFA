
from __future__ import annotations

from cqfa.group import get_representation

ALL4 = ["cqfa", "classical_fa", "non_equivariant", "augmented"]
SWEEP_REPS = ["z2_rc", "z2_pauli2", "z2_pauli3", "z2_pauli4", "c4", "c8", "d4"]
SPT_REPS = ["z2_pauli2", "z2_pauli3", "z2_pauli4"]

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
