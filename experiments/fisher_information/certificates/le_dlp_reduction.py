"""
experiments/fisher_information/certificates/le_dlp_reduction.py
===============================================
LEG LE  —  Phase-estimation -> discrete-log REDUCTION certificate (CONDITIONAL hardness).

We DO NOT prove any hardness. We certify the two *unconditional* arithmetic facts that
make the reduction rigorous, and demonstrate the decode:

  (A) PHASE LAW (unconditional, exact).  For G=Z_N (shift rep U_g|x>=|x+g mod N>), base
      state phi, concept psi_s = U_s phi, the Mode-F frame-Fourier component
          v_k(psi) = (1/N) sum_g w^{-kg} U_g psi ,   w = exp(-2*pi*i/N)
      satisfies  v_k(psi_s) = w^{k s} v_k(phi),  hence the cross-reference phase
          theta_k := arg < v_k(phi), v_k(psi_s) >  =  -2*pi*k*s/N   (mod 2*pi).
      VERIFY numeric==analytic to ~1e-12 for a range of prime N and all s.

  (B) DECODE (unconditional).  From theta_1 = -2*pi*s/N the hidden shift is
          s_hat = round( -theta_1 * N / (2*pi) )  mod N .
      Correct rounding is guaranteed whenever the phase error |dtheta_1| < pi/N
      (half the 2*pi/N spacing between consecutive s).  VERIFY decode success = 1.0 at
      an adversarial error just below pi/N, and that just above pi/N it can fail
      (threshold is tight).

  (C) DLP TIE (demonstration, NOT a break).  For a genuine discrete-log instance
      (p, g, x=g^s mod p) with g of prime order N in Z_p^*, the CQFA hidden shift s IS
      the discrete log dlog_g(x).  The decode recovers s from the phase, and we confirm
      g^{s_hat} == x (mod p).  => estimating theta_1 to precision < pi/N solves DLP.
      Therefore classical phase estimation is DLP-hard CONDITIONAL on DLP hardness
      (Liu-Arunachalam-Temme 2021 / abelian-HSP).  This code makes NO hardness claim.

Output CSV: results/fisher_information/certificates/le_dlp_reduction.csv
Pure numpy, deterministic per seed. Runs in seconds.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np


# ---------------------------------------------------------------------------
# Mode-F frame-Fourier component (matches experiments/fisher_information/fourier_mode.py)
# ---------------------------------------------------------------------------
def shift(N: int, g: int) -> np.ndarray:
    return np.roll(np.eye(N, dtype=complex), g, axis=0)          # U_g|x> = |x+g mod N>


def mode_F_components(psi: np.ndarray, N: int):
    """v_k = (1/N) sum_g w^{-kg} U_g psi,  w = exp(-2*pi*i/N)."""
    w = np.exp(-2j * np.pi / N)
    comps = []
    for k in range(N):
        v = sum((w ** (-k * g)) * (shift(N, g) @ psi) for g in range(N)) / N
        comps.append(v)
    return comps


def rand_state(N, rng):
    v = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# number theory helpers for the genuine DLP instances
# ---------------------------------------------------------------------------
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def prime_order_generator(p: int, q: int, rng):
    """Return g in Z_p^* of exact prime order q, where p = 2q+1 (safe prime)."""
    for _ in range(1000):
        h = int(rng.integers(2, p - 1))
        g = pow(h, 2, p)                 # squares have order dividing q (prime)
        if g != 1 and pow(g, q, p) == 1:
            return g
    raise RuntimeError("no generator found")


# ---------------------------------------------------------------------------
# (A)+(B): pure CQFA phase law + decode, full sweep over prime N and all s
# ---------------------------------------------------------------------------
def certify_phase_and_decode(N, seeds):
    max_phase_err = 0.0            # numeric vs analytic theta_k, all k with weight, all s
    decodes = 0
    total = 0
    min_success_margin = np.inf   # smallest gap (pi/N) - |used error| over successful decodes
    fail_above = False            # decode fails when error slightly exceeds pi/N
    two_pi = 2 * np.pi
    spacing_half = np.pi / N      # = tight decode threshold
    for sd in seeds:
        rng = np.random.default_rng(4242 + sd)
        # generic base state; for prime N every k!=0 has (generically) nonzero weight
        phi = rand_state(N, rng)
        ref = mode_F_components(phi, N)
        wt = np.array([float(np.real(np.vdot(ref[k], ref[k]))) for k in range(N)])
        for s in range(N):
            psi = shift(N, s) @ phi
            comps = mode_F_components(psi, N)
            for k in range(N):
                if wt[k] < 1e-9:
                    continue
                ov = np.vdot(ref[k], comps[k])          # = w^{k s} ||v_k(phi)||^2
                theta = float(np.angle(ov))
                analytic = -two_pi * k * s / N
                d = (theta - analytic) % two_pi
                d = min(d, two_pi - d)                  # wrapped abs phase error
                max_phase_err = max(max_phase_err, d)
            # decode from k=1
            theta1 = float(np.angle(np.vdot(ref[1], comps[1])))
            # (B) adversarial error just BELOW pi/N -> must still decode correctly
            err = 0.499 * two_pi / N                    # < pi/N
            sign = 1.0 if (s % 2 == 0) else -1.0
            s_hat = int(round(-(theta1 + sign * err) * N / two_pi)) % N
            total += 1
            if s_hat == s:
                decodes += 1
                min_success_margin = min(min_success_margin, spacing_half - err)
            # error just ABOVE pi/N pushed toward the midpoint -> can misdecode
            err_bad = 0.501 * two_pi / N
            s_bad = int(round(-(theta1 + sign * err_bad) * N / two_pi)) % N
            if s_bad != s:
                fail_above = True
    return max_phase_err, decodes, total, float(min_success_margin), fail_above


# ---------------------------------------------------------------------------
# (C): genuine discrete-log instances -> phase decodes the discrete log
# ---------------------------------------------------------------------------
def certify_dlp(safe_primes, seeds):
    rows = []
    two_pi = 2 * np.pi
    for (p, q) in safe_primes:            # p = 2q+1, q prime = order of g
        N = q
        ok_all = True
        max_err = 0.0
        n_inst = 0
        for sd in seeds:
            rng = np.random.default_rng(9000 + sd + p)
            g = prime_order_generator(p, q, rng)
            phi = rand_state(N, rng)
            ref = mode_F_components(phi, N)
            s = int(rng.integers(1, q))       # secret discrete log
            x = pow(g, s, p)                  # public element x = g^s mod p
            # CQFA concept for this DLP instance: hidden shift = s = dlog_g(x)
            psi = shift(N, s) @ phi
            comps = mode_F_components(psi, N)
            theta1 = float(np.angle(np.vdot(ref[1], comps[1])))
            # confirm phase law exactly
            analytic = (-two_pi * 1 * s / N)
            d = (theta1 - analytic) % two_pi
            d = min(d, two_pi - d)
            max_err = max(max_err, d)
            # decode the discrete log from the phase (precision pi/N, exact here)
            s_hat = int(round(-theta1 * N / two_pi)) % N
            # DLP verification: is the decoded exponent a true discrete log of x?
            recovered_ok = (pow(g, s_hat, p) == x) and (s_hat == s)
            ok_all = ok_all and recovered_ok
            n_inst += 1
        rows.append(dict(p=p, q=q, N=N, n_instances=n_inst,
                         max_phase_err=max_err,
                         dlp_decode_success=1.0 if ok_all else 0.0))
        print(f"  DLP  p={p:>4} q(order)={q:>4}: {n_inst} instances  "
              f"max|theta-analytic|={max_err:.2e}  "
              f"discrete-log recovered from phase: {'YES (1.0)' if ok_all else 'NO'}")
    return rows


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(primes, seeds, out_path):
    print("=" * 74)
    print("LEG LE  —  phase -> discrete-log REDUCTION certificate")
    print("=" * 74)
    print("(A)+(B) PHASE LAW theta_k = -2*pi*k*s/N  and DECODE at precision pi/N")
    print("-" * 74)
    rows = []
    global_max_err = 0.0
    total_dec = 0
    total_all = 0
    global_margin = np.inf
    all_fail_above = True
    for N in primes:
        me, dec, tot, margin, fail_above = certify_phase_and_decode(N, seeds)
        global_max_err = max(global_max_err, me)
        total_dec += dec
        total_all += tot
        global_margin = min(global_margin, margin)
        all_fail_above = all_fail_above and fail_above
        print(f"  N={N:>3} (prime): all-k max|theta-analytic|={me:.2e}   "
              f"decode {dec}/{tot} at err<pi/N   "
              f"threshold-tight(fails just above pi/N)={fail_above}")
        rows.append(dict(
            part="phase+decode", N=N, prime_order=True,
            max_phase_err=me, decode_success=dec / tot,
            n_decodes=tot, decode_err_used="0.499*2pi/N",
            threshold_tight=fail_above,
            p="", q="", dlp_decode_success="",
        ))

    print()
    print("(C) GENUINE DISCRETE-LOG INSTANCES  (p=2q+1 safe prime, g of order q)")
    print("-" * 74)
    # build safe-prime list: q prime, p=2q+1 prime
    safe = []
    q = 5
    while len(safe) < 8:
        if is_prime(q) and is_prime(2 * q + 1):
            safe.append((2 * q + 1, q))
        q += 1
    dlp_rows = certify_dlp(safe, seeds)
    dlp_all_ok = all(r["dlp_decode_success"] == 1.0 for r in dlp_rows)
    for r in dlp_rows:
        rows.append(dict(
            part="dlp", N=r["N"], prime_order=True,
            max_phase_err=r["max_phase_err"], decode_success="",
            n_decodes=r["n_instances"], decode_err_used="0 (exact)",
            threshold_tight="", p=r["p"], q=r["q"],
            dlp_decode_success=r["dlp_decode_success"],
        ))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    print("=" * 74)
    print("CERTIFICATE SUMMARY  (LEG LE)")
    print("=" * 74)
    print(f"  (A) phase law     : max over all (N,s,k) |theta_num - (-2pi k s/N)| "
          f"= {global_max_err:.2e}  (threshold 1e-12 -> "
          f"{'PASS' if global_max_err < 1e-12 else 'FAIL'})")
    print(f"  (B) decode        : {total_dec}/{total_all} correct at err=0.499*2pi/N "
          f"(< pi/N)  -> {'PASS' if total_dec == total_all else 'FAIL'}; "
          f"threshold tight (fails just above pi/N on every N): {all_fail_above}")
    print(f"  (C) DLP tie       : discrete log recovered from phase on all "
          f"safe-prime instances -> {'PASS' if dlp_all_ok else 'FAIL'}")
    print()
    print("  REDUCTION (conditional): classical estimation of theta_1 to precision")
    print("  < pi/N  outputs s = dlog_g(x)  => DLP-hard CONDITIONAL on DLP hardness")
    print("  (Liu-Arunachalam-Temme 2021 / abelian-HSP).  No hardness is proven here.")
    print("wrote", out_path)
    return rows, global_max_err, (total_dec == total_all), dlp_all_ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--primes", type=int, nargs="+",
                    default=[5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/fisher_information/certificates/le_dlp_reduction.csv")
    args = ap.parse_args(argv)
    run(args.primes, args.seeds, args.out)


if __name__ == "__main__":
    main()
