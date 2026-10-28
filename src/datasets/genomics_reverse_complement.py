
from __future__ import annotations

import numpy as np

from cqfa.group import Representation, close_matrices

# ---------------------------------------------------------------------------
# alphabet + reverse-complement primitives
# ---------------------------------------------------------------------------
BASES = ("A", "C", "G", "T")
BASE_IDX = {b: i for i, b in enumerate(BASES)}
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}
# index-level complement permutation: A(0)<->T(3), C(1)<->G(2)
COMP_IDX = np.array([BASE_IDX[COMPLEMENT[b]] for b in BASES])   # [3,2,1,0]


def reverse_complement(seq: str) -> str:
    """RC of a DNA string: reverse order and complement each base."""
    return "".join(COMPLEMENT[b] for b in reversed(seq))


# ---------------------------------------------------------------------------
# k-mer <-> amplitude-index encoding
# ---------------------------------------------------------------------------
def kmer_to_index(kmer: str) -> int:
    """Base-4 index of a length-k k-mer (A=0,C=1,G=2,T=3), MSB = first base."""
    idx = 0
    for b in kmer:
        idx = idx * 4 + BASE_IDX[b]
    return idx


def index_to_kmer(idx: int, k: int) -> str:
    digits = []
    for _ in range(k):
        digits.append(BASES[idx % 4])
        idx //= 4
    return "".join(reversed(digits))


def onehot_state(kmer: str) -> np.ndarray:
    """Unit-norm one-hot amplitude vector for a k-mer over 4^k basis states."""
    k = len(kmer)
    v = np.zeros(4 ** k, dtype=complex)
    v[kmer_to_index(kmer)] = 1.0
    return v


# ---------------------------------------------------------------------------
# the U_RC representation (group action == reverse complement)
# ---------------------------------------------------------------------------
def rc_permutation(k: int) -> np.ndarray:
    """Index permutation perm[i] = index of RC(kmer_i) over the 4^k basis.

    RC reverses base order AND complements each base. On the one-hot amplitude
    space this is a pure permutation of the 4^k computational-basis states.
    """
    n = 4 ** k
    perm = np.empty(n, dtype=int)
    for i in range(n):
        km = index_to_kmer(i, k)
        perm[i] = kmer_to_index(reverse_complement(km))
    return perm


def rc_representation(k: int = 2) -> Representation:
    """Z2 representation {I, U_RC} whose non-trivial element is the RC involution.

    U_RC is the permutation matrix of `rc_permutation(k)`. We assert it is an
    involution (U_RC @ U_RC == I) and close it into a Z2 of order 2 so the whole
    cqfa.readout machinery runs unmodified.
    """
    n = 4 ** k
    perm = rc_permutation(k)
    U = np.zeros((n, n), dtype=complex)
    for i, pi in enumerate(perm):
        U[pi, i] = 1.0
    assert np.allclose(U @ U, np.eye(n), atol=1e-12), "U_RC is not an involution"
    mats = close_matrices([U], n)
    assert len(mats) == 2, f"RC group must be Z2 (order 2), got {len(mats)}"
    return Representation(f"RC-k{k}", mats, n)


# ---------------------------------------------------------------------------
# synthetic RC-symmetric generator (source B, default)
# ---------------------------------------------------------------------------
def _rand_seq(length: int, rng: np.random.Generator) -> str:
    return "".join(BASES[i] for i in rng.integers(0, 4, size=length))


def _contains_motif_rc_invariant(seq: str, motif: str) -> bool:
    """True iff the motif OR its reverse complement occurs in seq.

    Membership defined on {seq} U {RC(seq)} makes the concept RC-invariant:
    a substring m occurs in seq  <=>  RC(m) occurs in RC(seq), so the predicate
    is unchanged under seq -> RC(seq). (Verified per-batch in _verify_rc_invariance.)
    """
    return (motif in seq) or (reverse_complement(motif) in seq)


def _flip_noise(seq: str, rho: float, rng: np.random.Generator) -> str:
    """Site-level i.i.d. base substitution at rate rho (makes task non-separable)."""
    out = []
    for b in seq:
        if rng.random() < rho:
            out.append(BASES[rng.integers(0, 4)])
        else:
            out.append(b)
    return "".join(out)


def generate_sequences(n_samples: int, rng: np.random.Generator,
                       seq_len: int = 12, motif: str = "CG", rho: float = 0.1):
    """Generate (sequences, labels) with an RC-invariant motif-presence concept.

    Class 1 (positive): sequences that (after noise) contain motif OR RC(motif).
    Class 0 (negative): sequences that contain neither.
    We rejection-sample to a balanced set, apply base-flip noise at rate rho, and
    re-derive the label from the noisy sequence, so labels stay RC-invariant and
    the task is not perfectly separable.

    Returns (list[str] sequences, np.ndarray[int] labels).
    """
    seqs, labels = [], []
    n_pos = n_samples // 2
    n0 = n1 = 0
    max_tries = 500 * n_samples + 10000
    tries = 0
    while (n1 < n_pos or n0 < n_samples - n_pos) and tries < max_tries:
        tries += 1
        raw = _rand_seq(seq_len, rng)
        noisy = _flip_noise(raw, rho, rng)
        lab = 1 if _contains_motif_rc_invariant(noisy, motif) else 0
        if lab == 1 and n1 < n_pos:
            seqs.append(noisy); labels.append(1); n1 += 1
        elif lab == 0 and n0 < n_samples - n_pos:
            seqs.append(noisy); labels.append(0); n0 += 1
    return seqs, np.array(labels, dtype=int)


def _verify_rc_invariance(seqs, labels, motif: str) -> None:
    """Invariance check: y(seq) == y(RC(seq)) for every row.

    The label is a function of the RC-invariant predicate on the (already noisy)
    sequence, so recomputing the predicate on RC(seq) must return the same label.
    """
    for seq, y in zip(seqs, labels):
        y_rc = 1 if _contains_motif_rc_invariant(reverse_complement(seq), motif) else 0
        y_self = 1 if _contains_motif_rc_invariant(seq, motif) else 0
        assert y_self == int(y), "stored label disagrees with the RC-invariant predicate"
        assert y_rc == int(y), "label is NOT reverse-complement invariant"


# ---------------------------------------------------------------------------
# window -> quantum-state encoding (RC of a WINDOW == group action on the state)
# ---------------------------------------------------------------------------
def _window_to_state(seq: str, k: int, rng: np.random.Generator) -> np.ndarray:
 
    counts = np.zeros(4 ** k, dtype=float)
    for i in range(len(seq) - k + 1):
        counts[kmer_to_index(seq[i:i + k])] += 1.0
    amp = np.sqrt(counts)
    nrm = np.linalg.norm(amp)
    if nrm < 1e-12:                                  # degenerate; fall back uniform
        amp = np.ones(4 ** k)
        nrm = np.linalg.norm(amp)
    return (amp / nrm).astype(complex)


def encode_dataset(seqs, k: int, rng: np.random.Generator) -> np.ndarray:
    """Encode a list of sequences into an (N, 4^k) complex unit-state array."""
    return np.array([_window_to_state(s, k, rng) for s in seqs])


# ---------------------------------------------------------------------------
# public builder: (X, y) + Representation for the sweep
# ---------------------------------------------------------------------------
def build_genomics_rc(n_samples: int, rng: np.random.Generator, k: int = 2,
                      seq_len: int = 12, motif: str = "CG", rho: float = 0.1):

    seqs, y = generate_sequences(n_samples, rng, seq_len=seq_len, motif=motif, rho=rho)
    _verify_rc_invariance(seqs, y, motif)             # invariance gate
    X = encode_dataset(seqs, k, rng)
    rep = rc_representation(k)
    meta = dict(dataset_source="synthetic", k=k, seq_len=seq_len, motif=motif,
                rho=rho, n_samples=len(seqs))
    return X, y, rep, meta, seqs


def make_task(k: int = 2, seq_len: int = 12, motif: str = "CG", rho: float = 0.1):
    """Return a closure (n_samples, rng) -> (X, y) for the sweep driver.

    The rep is fixed by k; it is exposed via `.rep` on the returned closure so the
    driver can pass it to run_experiment(rep=...).
    """
    rep = rc_representation(k)

    def task_fn(n_samples, rng):
        seqs, y = generate_sequences(n_samples, rng, seq_len=seq_len, motif=motif, rho=rho)
        _verify_rc_invariance(seqs, y, motif)
        X = encode_dataset(seqs, k, rng)
        return X, y

    task_fn.rep = rep
    task_fn.k = k
    return task_fn


if __name__ == "__main__":
    # tiny self-check
    rng = np.random.default_rng(0)
    for k in (1, 2):
        rep = rc_representation(k)
        print(f"k={k}: rep dim={rep.dim} order={rep.order} "
              f"closed={rep.is_closed()} unitary={rep.is_unitary()}")
        X, y, rep, meta, seqs = build_genomics_rc(60, np.random.default_rng(1),
                                                  k=k, seq_len=12)
        print(f"   X={X.shape} balance={y.mean():.2f} unitnorm="
              f"{np.allclose(np.linalg.norm(X, axis=1), 1.0)} source={meta['dataset_source']}")
