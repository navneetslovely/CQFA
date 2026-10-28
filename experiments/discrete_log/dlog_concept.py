
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# elementary number theory (small primes, brute force -- scope)
# ---------------------------------------------------------------------------
def is_prime(n: int) -> bool:
    """Trial-division primality (fine for the small primes used here)."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _prime_factors(m: int) -> list[int]:
    """Distinct prime factors of m (m >= 1)."""
    factors, d = [], 2
    while d * d <= m:
        if m % d == 0:
            factors.append(d)
            while m % d == 0:
                m //= d
        d += 1
    if m > 1:
        factors.append(m)
    return factors


def is_primitive_root(g: int, p: int) -> bool:
    """True iff g generates Z_p^* (order of g modulo p equals p-1)."""
    if g % p == 0:
        return False
    order = p - 1
    for q in _prime_factors(order):
        if pow(g, order // q, p) == 1:
            return False
    return True


def smallest_primitive_root(p: int) -> int:
    """Smallest primitive root g of the prime p (deterministic canonical choice)."""
    if not is_prime(p):
        raise ValueError(f"p={p} is not prime")
    for g in range(2, p):
        if is_primitive_root(g, p):
            return g
    raise ValueError(f"no primitive root found for p={p} (should be impossible for a prime)")


def discrete_log_table(g: int, p: int) -> dict[int, int]:
    """Brute-force table  x -> log_g(x)  for every x in Z_p^* = {1,...,p-1}.

    Built by iterating powers of g (O(p)); valid because g is a primitive root,
    so {g**0, ..., g**(p-2)} enumerates Z_p^* exactly once.
    """
    table, val = {}, 1
    for e in range(p - 1):
        table[val] = e
        val = (val * g) % p
    assert len(table) == p - 1, "generator did not enumerate all of Z_p^*"
    return table


# ---------------------------------------------------------------------------
# the concept class
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DlogConcept:
    """The LAT discrete-log concept class over Z_p^* with generator g.

    Attributes
    ----------
    p     : the prime.
    g     : a primitive root modulo p (generator of Z_p^*).
    logs  : dict  x -> log_g(x)  (brute-forced, small p only).
    order : |Z_p^*| = p-1 (the order of the cyclic log group C_{p-1}).
    """

    p: int
    g: int
    logs: dict
    order: int

    @classmethod
    def build(cls, p: int, g: int | None = None) -> "DlogConcept":
        if not is_prime(p):
            raise ValueError(f"p={p} must be prime")
        if p == 2:
            raise ValueError("p must be an odd prime (need (p-3)/2 to be an integer half-arc)")
        if g is None:
            g = smallest_primitive_root(p)
        elif not is_primitive_root(g, p):
            raise ValueError(f"g={g} is not a primitive root mod p={p}")
        return cls(p=p, g=g, logs=discrete_log_table(g, p), order=p - 1)

    # -- the concept ---------------------------------------------------------
    def label(self, x: int, s: int) -> int:
        """f_s(x) in {+1,-1}: half-interval-in-log membership for secret s.

        +1 iff (log_g(x) - s) mod (p-1) lies in {0,...,(p-3)/2}, i.e. the lower
        half-arc of the cyclic log coordinate rooted at s.
        """
        half = (self.p - 3) // 2  # (p-1)/2 - 1 ; arc length = half+1 = (p-1)/2
        d = (self.logs[x] - s) % self.order
        return 1 if d <= half else -1

    def labels(self, xs: np.ndarray, s: int) -> np.ndarray:
        return np.array([self.label(int(x), s) for x in xs], dtype=int)

    def domain(self) -> np.ndarray:
        """The full data domain Z_p^* = {1,...,p-1} (sorted, deterministic)."""
        return np.arange(1, self.p, dtype=int)


# ---------------------------------------------------------------------------
# dataset generation (deterministic; all randomness via a passed Generator)
# ---------------------------------------------------------------------------
def make_dataset(concept: DlogConcept, s: int, n_train: int, n_test: int,
                 rng: np.random.Generator):
    """Sample a train/test split of (x, y=f_s(x)) without replacement.

    Returns (x_train, y_train, x_test, y_test) with x's in Z_p^* and y in {+1,-1}.
    The train and test x-sets are disjoint (true generalisation, no leakage).
    """
    domain = concept.domain()
    if n_train + n_test > len(domain):
        raise ValueError(
            f"n_train+n_test={n_train + n_test} exceeds |Z_p^*|={len(domain)} for p={concept.p}"
        )
    perm = rng.permutation(domain)
    x_train = perm[:n_train]
    x_test = perm[n_train:n_train + n_test]
    y_train = concept.labels(x_train, s)
    y_test = concept.labels(x_test, s)
    return x_train, y_train, x_test, y_test


def random_secret(concept: DlogConcept, rng: np.random.Generator) -> int:
    """A uniformly random concept secret s in Z_{p-1}."""
    return int(rng.integers(0, concept.order))


# ---------------------------------------------------------------------------
# self-test (numbers are traceable; run: python -m experiments.discrete_log.dlog_concept)
# ---------------------------------------------------------------------------
def _selftest() -> bool:
    ok = True
    for p in (7, 11, 23, 47):
        c = DlogConcept.build(p)
        # generator correctness: powers of g enumerate Z_p^* exactly once
        assert set(c.logs.keys()) == set(range(1, p)), (p, "logs domain")
        assert is_primitive_root(c.g, p), (p, "g not primitive root")
        # concept is exactly balanced: half-arc has (p-1)/2 of the p-1 log values
        s = 0
        ys = c.labels(c.domain(), s)
        npos = int(np.sum(ys == 1))
        expected_pos = (p - 1) // 2
        balanced = (npos == expected_pos)
        # log-space monotone structure: labels are a contiguous arc in log coord
        by_log = np.array([c.label(x, s) for x in
                           sorted(c.domain(), key=lambda x: c.logs[x])])
        # count sign changes around the cyclic log axis -> exactly 2 (one arc)
        sign_changes = int(np.sum(by_log != np.roll(by_log, 1)))
        arc_ok = (sign_changes == 2)
        print(f"p={p:>3} g={c.g:>2}  |Z_p^*|={p - 1:>3}  "
              f"#pos={npos:>3}/{expected_pos:<3} balanced={balanced}  "
              f"log-arc sign-changes={sign_changes} (want 2) arc_ok={arc_ok}")
        ok = ok and balanced and arc_ok
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
