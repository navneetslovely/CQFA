"""Coherent Quantum Frame Averaging.

A finite group G acts on a data register through a unitary representation. A
frame register of ceil(log2 |G|) ancillas holds the uniform superposition over
group elements, and a controlled oracle applies the representation conditioned
on the frame. Measuring the frame selects the readout mode.

Modules
-------
group                   finite groups, unitary representations, group oracle
frame                   frame preparation, coherent projection, Modes A and B
readout                 the four matched-parameter model variants
simulator               statevector ansatz with parameter-shift gradients
dynamical_lie_algebra   Lie closure, restricted algebras, gradient variance
certificate             computable trainability and hardness components
tasks                   learning-task construction
train                   seeded training, bootstrap intervals, CSV logging
verify                  numerical checks of the invariance identities
paths                   filesystem locations for results, runs and figures

Pure NumPy and deterministic.
"""
from . import (  # noqa: F401
    certificate,
    dynamical_lie_algebra,
    frame,
    group,
    paths,
    readout,
    simulator,
    tasks,
    train,
    verify,
)

__all__ = [
    "certificate",
    "dynamical_lie_algebra",
    "frame",
    "group",
    "paths",
    "readout",
    "simulator",
    "tasks",
    "train",
    "verify",
]
__version__ = "1.0.0"
