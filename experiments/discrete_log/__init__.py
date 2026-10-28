"""
experiments.discrete_log
===================
The Liu--Arunachalam--Temme discrete-log
concept-class proof-of-principle.

A small-prime instantiation showing that a learner AWARE of the cyclic-log /
group structure (the covariant C_{p-1} / CQFA quantum kernel) separates the
half-interval-in-log concept from few samples, while a strong CLASSICAL learner
on raw x features does not generalise. Scope: proof-of-principle at
simulable p; NO scaling claim; the asymptotic classical hardness is the LAT
theorem under the discrete-logarithm assumption (CITED, not re-proven here).

Modules
-------
* dlog_concept : Z_p^*, primitive roots, brute-force discrete logs, the concept f_s,
                 and dataset generation.
* learners     : the covariant-C_{p-1} kernel (structured) and the raw-x classical
                 control, with train/eval helpers.
* run_sweep       : the multi-seed driver (mean +/- std, bootstrap 95% CIs, CSVs).
"""
from .dlog_concept import (
    DlogConcept,
    is_prime,
    is_primitive_root,
    smallest_primitive_root,
    discrete_log_table,
    make_dataset,
    random_secret,
)
from .learners import (
    fiducial_state,
    covariant_states,
    covariant_kernel,
    raw_x_features,
    structured_accuracy,
    classical_raw_accuracy,
    chance_accuracy,
)
from .run_sweep import run

__all__ = [
    "DlogConcept",
    "is_prime",
    "is_primitive_root",
    "smallest_primitive_root",
    "discrete_log_table",
    "make_dataset",
    "random_secret",
    "fiducial_state",
    "covariant_states",
    "covariant_kernel",
    "raw_x_features",
    "structured_accuracy",
    "classical_raw_accuracy",
    "chance_accuracy",
    "run",
]
