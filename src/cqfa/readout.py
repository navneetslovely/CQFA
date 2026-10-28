"""The four model variants, sharing one ansatz and one observable.

All four have identical parameter counts, so comparisons between them are
controlled. They differ only in how the group frame and orbit are handled:

  CQFAModel          Mode A. Coherent projection Pi psi, then trainable W,
                     then read O. Exactly G-invariant for every theta, and
                     sensitive to cross-orbit interference.
  ClassicalFAModel   Mode B. Average the orbit outputs of the same W and O.
                     Exactly G-invariant, without interference.
  NonEquivModel      No frame. Trainable W on the raw state. Not invariant.
  AugmentedModel     NonEquiv forward pass trained with orbit data
                     augmentation, giving invariance by data rather than by
                     architecture.

The decision feature f(x; theta) is an expectation value in [-1, 1].
"""
from __future__ import annotations

import numpy as np

from .group import Representation
from .simulator import Ansatz, local_z
from . import frame as L


def _n_qubits_for(dim: int) -> int:
    n = int(round(np.log2(dim)))
    if 2 ** n != dim:
        raise ValueError(f"trainable models need a power-of-two rep dim, got dim={dim}")
    return n


class Model:
    """Base: shared ansatz + observable; subclasses define `feature`."""

    equivariant = True
    augmented = False

    def __init__(self, rep: Representation, n_layers: int = 3, observable=None, name="model"):
        self.rep = rep
        self.n_qubits = _n_qubits_for(rep.dim)
        self.ansatz = Ansatz(self.n_qubits, n_layers)
        self.observable = observable if observable is not None else local_z(self.n_qubits, 0)
        self.n_params = self.ansatz.n_params
        self.name = name

    def init_params(self, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal(self.n_params) * 0.1

    def feature(self, theta: np.ndarray, psi: np.ndarray):
        """Return (f, df/dtheta): the decision feature and its parameter gradient."""
        raise NotImplementedError

    def features_batch(self, theta: np.ndarray, X: np.ndarray):
        """Batched (fs (N,), grads (N,P)) over rows of X. Overridden per variant."""
        raise NotImplementedError


class CQFAModel(Model):
    equivariant = True

    def __init__(self, rep, n_layers=3, observable=None):
        super().__init__(rep, n_layers, observable, name="cqfa")

    def feature(self, theta, psi):
        phi, p = L.coherent_projection(psi, self.rep)
        if p < 1e-12:                      # annihilated by Pi: the feature is zero
            return 0.0, np.zeros(self.n_params)
        chi = phi / np.sqrt(p)             # normalised coherent projection (theta-independent)
        return self.ansatz.expval_grad(theta, chi, self.observable)

    def features_batch(self, theta, X):
        Phi = X @ self.rep.projector().T                 # Pi applied to each row
        p = np.real(np.einsum("ni,ni->n", Phi.conj(), Phi))
        Xchi = np.zeros_like(Phi)
        ok = p >= 1e-12
        Xchi[ok] = Phi[ok] / np.sqrt(p[ok])[:, None]
        return self.ansatz.expvals_grad(theta, Xchi, self.observable)


class ClassicalFAModel(Model):
    equivariant = True

    def __init__(self, rep, n_layers=3, observable=None):
        super().__init__(rep, n_layers, observable, name="classical_fa")

    def feature(self, theta, psi):
        vals, grads = [], []
        for U in self.rep.matrices:
            v, g = self.ansatz.expval_grad(theta, U @ psi, self.observable)
            vals.append(v)
            grads.append(g)
        return float(np.mean(vals)), np.mean(grads, axis=0)

    def features_batch(self, theta, X):
        N, P = X.shape[0], self.n_params
        vals = np.zeros(N)
        grads = np.zeros((N, P))
        for U in self.rep.matrices:
            v, g = self.ansatz.expvals_grad(theta, X @ U.T, self.observable)
            vals += v
            grads += g
        return vals / self.rep.order, grads / self.rep.order


class NonEquivModel(Model):
    equivariant = False

    def __init__(self, rep, n_layers=3, observable=None):
        super().__init__(rep, n_layers, observable, name="non_equivariant")

    def feature(self, theta, psi):
        return self.ansatz.expval_grad(theta, psi, self.observable)

    def features_batch(self, theta, X):
        return self.ansatz.expvals_grad(theta, X, self.observable)


class AugmentedModel(NonEquivModel):
    """Same forward as NonEquiv; train.py augments each sample with its orbit."""

    equivariant = False
    augmented = True

    def __init__(self, rep, n_layers=3, observable=None):
        Model.__init__(self, rep, n_layers, observable, name="augmented")


def build_models(rep: Representation, n_layers: int = 3) -> dict:
    """The four matched-parameter variants for a given representation."""
    return {
        "cqfa": CQFAModel(rep, n_layers),
        "classical_fa": ClassicalFAModel(rep, n_layers),
        "non_equivariant": NonEquivModel(rep, n_layers),
        "augmented": AugmentedModel(rep, n_layers),
    }
