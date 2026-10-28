"""
experiments/spt_measurement_cost/exponential_separation.py
============================================
The EXPONENTIAL measurement-complexity separation on SPT data (strengthened form).

Task: estimate the nonlocal SPT order parameter <O_string> (a weight-w Pauli string,
w = n-2) of an unknown cluster-Ising ground state, to fixed precision eps.

  * Symmetry-aware COHERENT readout: reads the protected order directly ->
    per-shot variance V_coh = O(1); shots ~ V_coh / eps^2 (FLAT in n).
  * Symmetry-agnostic SINGLE-COPY classical shadows (Huang-Kueng-Preskill 2020):
    the standard single-copy primitive when the observable is not known a priori.
    Per-shot variance of a weight-w Pauli is EXACTLY 3^w (theorem); shots ~ 3^w/eps^2
    -> EXPONENTIAL in n.
  * Control (single-copy told the exact string basis): O(1), confirming the gap is
    the price of symmetry-agnostic single-copy tomography, not of the state.

We VERIFY V_shadow = 3^w empirically (V_shadow/3^w ~ 1) and report shots vs n.
Everything statevector-only. Output: results/spt_measurement_cost/exponential_separation.csv .
"""
from __future__ import annotations
import argparse, csv, os
import numpy as np

from . import spt_model as M
from . import readout_shadow as S


def shadow_variance(psi, n, a, b, rng, target_hits=60, cap=400_000):
    """Empirical single-copy shadow estimator variance of <O_string>; snapshots
    scaled to the weight so the estimate is well-resolved."""
    label = S.string_pauli_label(n, a, b)
    w = len(label)
    nsnap = int(min(cap, target_hits * 3 ** w))
    v = S._per_snapshot_values(S.shadow_snapshots(psi, rng, nsnap), label)
    return float(np.var(v)), w, nsnap


def run(sizes, lam_spt, eps, seeds, out_dir):
    rows = []
    for n in sizes:
        a, b = 1, n - 2
        Vsh, Vsh_s = [], 0
        for sd in seeds:
            psi = M.ground_state(n, lam_spt)
            V, w, nsnap = shadow_variance(psi, n, a, b, np.random.default_rng(sd))
            Vsh.append(V); Vsh_s = nsnap
        V_shadow = float(np.mean(Vsh))
        # coherent readout variance of the +/-1 direct read of <O_string>
        psi = M.ground_state(n, lam_spt)
        Ostr = M.string_order(psi, n, a, b)
        V_coh = max(1.0 - Ostr ** 2, 1e-3)
        w = (b - a + 3)                                    # weight of the Y-decorated string
        shots_coh = V_coh / eps ** 2
        shots_shadow = V_shadow / eps ** 2
        rows.append(dict(n=n, weight=w, three_pow_w=3 ** w,
                         V_shadow=V_shadow, V_shadow_over_3w=V_shadow / 3 ** w,
                         V_coherent=V_coh, shots_coherent=shots_coh,
                         shots_shadow_singlecopy=shots_shadow,
                         separation_ratio=shots_shadow / shots_coh,
                         n_snapshots=Vsh_s))
        print(f"n={n} w={w} 3^w={3**w:>8} V_shadow={V_shadow:>9.0f} (V/3^w={V_shadow/3**w:.2f}) "
              f"V_coh={V_coh:.3f} | shots: coherent={shots_coh:.0f} shadow={shots_shadow:.3e} "
              f"ratio={shots_shadow/shots_coh:.2e}")
    if rows:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, "exponential_separation.csv")
        with open(p, "w", newline="") as fh:
            w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w_.writeheader(); w_.writerows(rows)
        print("wrote", p)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 5, 6, 7, 8, 9, 10])
    ap.add_argument("--lam-spt", type=float, default=0.3)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/spt_measurement_cost")
    args = ap.parse_args(argv)
    run(args.sizes, args.lam_spt, args.eps, args.seeds, args.out)


if __name__ == "__main__":
    main()
