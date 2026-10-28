"""A classical reverse-complement-symmetric baseline.

On classical genomics data an RC-parameter-shared classical model is a strong
baseline and is expected to be competitive with the quantum models. It is
reported alongside them so the comparison on this task is complete.

Design
------
Features are the RC-symmetrised k-mer count histogram of a sequence,

    h_sym(seq) = h(seq) + h(RC(seq)),

which is invariant under the reverse-complement map by construction. A logistic
classifier is fitted on those features. The model is deterministic: a fixed
solver, a fixed iteration cap and a fixed seed.

The data seed convention matches the quantum runs, so the two arms see the same
sequences for a given seed.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .genomics_reverse_complement import (BASES, kmer_to_index, reverse_complement,
                          generate_sequences, _verify_rc_invariance)


def rc_symmetrized_features(seqs, k: int) -> np.ndarray:
    """RC-invariant feature: h(seq) + h(RC(seq)) over the 4^k k-mer histogram."""
    n = 4 ** k
    feats = np.zeros((len(seqs), n), dtype=float)
    for r, seq in enumerate(seqs):
        for s in (seq, reverse_complement(seq)):
            for i in range(len(s) - k + 1):
                feats[r, kmer_to_index(s[i:i + k])] += 1.0
    return feats


class RCCNNClassical:
    """RC-equivariant classical classifier (RCPS-style logistic head).

    `n_params` is reported for provenance (feature-dim + bias); it is NOT matched
    to the quantum models (different model class) and the master CSV flags this
    row with quantum_model=False.
    """

    def __init__(self, k: int = 2, C: float = 1.0, seed: int = 0):
        self.k = k
        self.C = C
        self.seed = seed
        self.clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)

    def fit(self, seqs, y):
        X = rc_symmetrized_features(seqs, self.k)
        self.clf.fit(X, y)
        self.n_params = int(X.shape[1] + 1)          # weights + bias (provenance)
        return self

    def predict(self, seqs):
        return self.clf.predict(rc_symmetrized_features(seqs, self.k))

    def predict_proba(self, seqs):
        return self.clf.predict_proba(rc_symmetrized_features(seqs, self.k))[:, 1]


def run_rc_baseline_seed(seed, n_train, n_test, k=2, seq_len=12, motif="CG", rho=0.1):
    """Train + evaluate the classical RC baseline for one data seed.

    Uses the SAME data seed convention as cqfa.train.run_experiment (1000 + seed)
    so the classical baseline sees the same sequences as the quantum models.
    Returns dict(seed, y_true, p_score, acc).
    """
    drng = np.random.default_rng(1000 + seed)
    seqs, y = generate_sequences(n_train + n_test, drng, seq_len=seq_len,
                                 motif=motif, rho=rho)
    _verify_rc_invariance(seqs, y, motif)
    str_tr, y_tr = seqs[:n_train], y[:n_train]
    str_te, y_te = seqs[n_train:], y[n_train:]
    model = RCCNNClassical(k=k, seed=seed).fit(str_tr, y_tr)
    p_te = model.predict_proba(str_te)
    acc = float(np.mean((p_te > 0.5).astype(int) == y_te))
    return dict(seed=int(seed), y_true=np.asarray(y_te, dtype=int),
                p_score=np.asarray(p_te, dtype=float), acc=acc,
                n_params=int(model.n_params))


if __name__ == "__main__":
    for k in (1, 2):
        r = run_rc_baseline_seed(0, 48, 32, k=k)
        print(f"k={k}: acc={r['acc']:.3f} n_params={r['n_params']}")
