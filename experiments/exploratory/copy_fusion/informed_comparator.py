
from __future__ import annotations

import argparse
import csv
import os
import numpy as np

# ---------------------------------------------------------------- Pauli algebra
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
_PAULI = {"X": X, "Y": Y, "Z": Z}
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_HSdg = _H @ np.array([[1, 0], [0, -1j]], dtype=complex)
# rotation taking a Pauli-basis measurement to a Z-basis measurement
_ROT = {"X": _H, "Y": _HSdg, "Z": np.eye(2, dtype=complex)}
_EIGV = {k: np.linalg.eigh(v)[1] for k, v in _PAULI.items()}  # columns = eigenvectors
_NAMES = ["X", "Y", "Z"]

EPS = 0.1


def _kron(ms):
    o = np.array([[1.0 + 0j]])
    for m in ms:
        o = np.kron(o, m)
    return o


def _pauli_string(n):
    """The weight-n string of thm:mc / e8_fusion: Y on site 0, X elsewhere."""
    return ["Y"] + ["X"] * (n - 1)


def _string_op(labels):
    return _kron([_PAULI[c] for c in labels])


# ================================================================ TASK A (linear)
def _state_with_expectation(P, n, target, rng):
    """|psi> with <psi|P|psi> = target exactly (target in (-1,1))."""
    d = 2 ** n
    v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    proj_p = (np.eye(d) + P) / 2.0
    proj_m = (np.eye(d) - P) / 2.0
    vp, vm = proj_p @ v, proj_m @ v
    vp /= np.linalg.norm(vp)
    vm /= np.linalg.norm(vm)
    a = (target + 1.0) / 2.0
    psi = np.sqrt(a) * vp + np.sqrt(1.0 - a) * vm
    return psi / np.linalg.norm(psi)


def _rotate(psi, n, bases):
    out = psi.reshape([2] * n)
    for q in range(n):
        out = np.moveaxis(np.tensordot(_ROT[bases[q]], out, axes=([1], [q])), 0, q)
    return out.reshape(-1)


def _informed_shot_probs(psi, n, labels):
    """Distribution over the 2^n outcomes when measuring in the eigenbases of P."""
    pr = np.abs(_rotate(psi, n, labels)) ** 2
    return pr / pr.sum()


def _parity_signs(n):
    idx = np.arange(2 ** n)
    bits = ((idx[:, None] >> np.arange(n - 1, -1, -1)[None, :]) & 1).sum(1)
    return 1.0 - 2.0 * (bits % 2)


def _rmse_informed_A(probs, signs, exact, M, R, rng):
    """M single-copy informed shots; estimator = mean of +-1 parities."""
    p_plus = float(probs[signs > 0].sum())
    k = rng.binomial(M, p_plus, size=R)
    est = (2.0 * k - M) / M
    return float(np.sqrt(np.mean((est - exact) ** 2)))


def _rmse_coherent_A(exact, M, R, rng):
    """Hadamard test on the coherence: P(0) = (1+ReC)/2, estimator +-1, 1 copy/shot."""
    p0 = (1.0 + exact) / 2.0
    k = rng.binomial(M, p0, size=R)
    est = (2.0 * k - M) / M
    return float(np.sqrt(np.mean((est - exact) ** 2)))


def _rmse_agnostic_A(exact, n, M, R, rng):
    """Random-Pauli shadows: estimator is 3^n * (+-1) on a basis hit (prob 3^-n), else 0."""
    hits = rng.binomial(M, 3.0 ** (-n), size=R)
    p_plus = (1.0 + exact) / 2.0
    plus = rng.binomial(hits, p_plus)
    est = (3.0 ** n) * (2.0 * plus - hits) / M
    return float(np.sqrt(np.mean((est - exact) ** 2)))


def _shadow_estimator_full_A(psi, n, labels, nshots, rng):
    """ basis-by-basis shadow simulation, used only to validate the fast path."""
    vals = np.empty(nshots)
    for s in range(nshots):
        bases = [_NAMES[b] for b in rng.integers(0, 3, n)]
        pr = np.abs(_rotate(psi, n, bases)) ** 2
        pr = np.clip(pr, 0, None)
        pr /= pr.sum()
        out = rng.choice(len(pr), p=pr)
        v = 1.0
        for q in range(n):
            if bases[q] != labels[q]:
                v = 0.0
                break
            bit = (out >> (n - 1 - q)) & 1
            v *= 3.0 * (1.0 - 2.0 * bit)
        vals[s] = v
    return vals


# =============================================================== TASK B (nonlinear)
def _mixed_state(psi, n, p):
    d = 2 ** n
    return (1 - p) * np.outer(psi, psi.conj()) + p * np.eye(d) / d


def _eigvals_depolarized(n, p):
    d = 2 ** n
    ev = np.full(d, p / d)
    ev[0] = (1 - p) + p / d
    return ev


def _rmse_informed_B(eigvals, exact, M, R, rng):
    """Informed: told the eigenbasis. M samples -> unbiased collision U-statistic."""
    if M < 2:
        return float("inf")
    out = np.empty(R)
    for r in range(R):
        c = rng.multinomial(M, eigvals)
        coll = float(np.sum(c * (c - 1)) / 2.0)
        out[r] = coll / (M * (M - 1) / 2.0)
    return float(np.sqrt(np.mean((out - exact) ** 2)))


def _rmse_coherent_B(exact, M, R, rng):
    """Two-copy SWAP test. M copies == M/2 shots (charged  at 2 copies/shot)."""
    shots = M // 2
    if shots < 1:
        return float("inf")
    p0 = (1.0 + exact) / 2.0
    k = rng.binomial(shots, p0, size=R)
    est = (2.0 * k - shots) / shots
    return float(np.sqrt(np.mean((est - exact) ** 2)))


def _snapshots(rho, n, m, rng):
    snaps = np.empty((m, 2 ** n, 2 ** n), dtype=complex)
    for s in range(m):
        bases = rng.integers(0, 3, n)
        U = _kron([_EIGV[_NAMES[b]].conj().T for b in bases])
        pr = np.real(np.diag(U @ rho @ U.conj().T))
        pr = np.clip(pr, 0, None)
        pr /= pr.sum()
        out = rng.choice(len(pr), p=pr)
        snap = np.array([[1.0 + 0j]])
        for q in range(n):
            bit = (out >> (n - 1 - q)) & 1
            v = _EIGV[_NAMES[bases[q]]][:, bit].reshape(-1, 1)
            snap = np.kron(snap, 3.0 * (v @ v.conj().T) - I2)
        snaps[s] = snap
    return snaps


def _rmse_agnostic_B(rho, n, exact, M, R, rng):
    """Shadow purity U-statistic; O(m) via Tr((sum S)^2) - sum Tr(S^2)."""
    if M < 2:
        return float("inf")
    out = np.empty(R)
    for r in range(R):
        S = _snapshots(rho, n, M, rng)
        tot = S.sum(0)
        cross = np.real(np.trace(tot @ tot)) - np.real(np.einsum("mij,mji->", S, S))
        out[r] = cross / (M * (M - 1))
    return float(np.sqrt(np.mean((out - exact) ** 2)))


# ================================================================ sweep utilities
def _copies_to_target(fn, grid, eps):
    """Smallest budget in the frozen grid with RMSE <= eps. Returns (value, lower_bound)."""
    for M in grid:
        if fn(M) <= eps:
            return M, ""
    return "", grid[-1]


def run(sizes_a, sizes_b, seeds, out_dir, matched_a=512, matched_b=256,
        reps_a=300, reps_b=60, target_exp=0.6, depol_p=0.5):
    grid_a = [2 ** k for k in range(2, 25)]
    grid_b_cheap = [2 ** k for k in range(3, 19)]
    grid_b_shadow = [2 ** k for k in range(3, 12)]  # cap 2048 snapshots (expensive)

    per_seed = []
    agg = []

    # -------------------------------------------------- TASK A: linear Pauli string
    for n in sizes_a:
        labels = _pauli_string(n)
        P = _string_op(labels)
        signs = _parity_signs(n)
        cells = []
        for sd in seeds:
            rng = np.random.default_rng(10_000 + 97 * sd + n)
            psi_p = _state_with_expectation(P, n, +target_exp, rng)
            exact = float(np.real(psi_p.conj() @ P @ psi_p))
            probs = _informed_shot_probs(psi_p, n, labels)
            signal = 2.0 * abs(target_exp)  # |<P>_+ - <P>_-| between the two classes

            r_inf = _rmse_informed_A(probs, signs, exact, matched_a, reps_a, rng)
            r_coh = _rmse_coherent_A(exact, matched_a, reps_a, rng)
            r_agn = _rmse_agnostic_A(exact, n, matched_a, reps_a, rng)

            c_inf, b_inf = _copies_to_target(
                lambda M: _rmse_informed_A(probs, signs, exact, M, 200, rng), grid_a, EPS)
            c_coh, b_coh = _copies_to_target(
                lambda M: _rmse_coherent_A(exact, M, 200, rng), grid_a, EPS)
            c_agn, b_agn = _copies_to_target(
                lambda M: _rmse_agnostic_A(exact, n, M, 200, rng), grid_a, EPS)

            # per-shot variances (empirical, 20000 shots) + closed forms for cross-check
            NV = 20000
            k = rng.binomial(1, float(probs[signs > 0].sum()), size=NV)
            v_inf = float(np.var(2.0 * k - 1.0, ddof=1))
            k = rng.binomial(1, (1.0 + exact) / 2.0, size=NV)
            v_coh = float(np.var(2.0 * k - 1.0, ddof=1))
            hit = rng.binomial(1, 3.0 ** (-n), size=NV)
            sg = 2.0 * rng.binomial(1, (1.0 + exact) / 2.0, size=NV) - 1.0
            v_agn = float(np.var((3.0 ** n) * hit * sg, ddof=1))

            # validate the fast shadow path against a full basis-by-basis simulation
            fp_err = ""
            if n <= 3:
                full = _shadow_estimator_full_A(psi_p, n, labels, 20000, rng)
                fp_err = abs(float(np.var(full, ddof=1)) - v_agn) / (3.0 ** n)

            cells.append(dict(rmse_informed=r_inf, rmse_coherent=r_coh, rmse_agnostic=r_agn))
            per_seed.append(dict(
                task="A_linear_pauli_string", n=n, seed=sd,
                exact_value=exact, label_signal_scale=signal,
                vacuous_excluded=int(signal < 1e-10),
                matched_copies=matched_a,
                rmse_agnostic=r_agn, rmse_informed=r_inf, rmse_coherent=r_coh,
                delta_informed_minus_coherent=r_inf - r_coh,
                var_per_shot_agnostic=v_agn, var_per_shot_informed=v_inf,
                var_per_shot_coherent=v_coh,
                var_agnostic_closed_form=3.0 ** n - exact ** 2,
                var_informed_closed_form=1.0 - exact ** 2,
                copies_to_target_agnostic=c_agn, copies_to_target_agnostic_lower_bound=b_agn,
                copies_to_target_informed=c_inf, copies_to_target_informed_lower_bound=b_inf,
                copies_to_target_coherent=c_coh, copies_to_target_coherent_lower_bound=b_coh,
                shadow_fastpath_check_relerr=fp_err))
        agg.append(_aggregate("A_linear_pauli_string", n, cells, per_seed, matched_a))
        a = agg[-1]
        print(f"[A] n={n}: RMSE agn={a['rmse_agnostic_mean']:.4f} "
              f"inf={a['rmse_informed_mean']:.4f} coh={a['rmse_coherent_mean']:.4f} | "
              f"delta={a['delta_informed_minus_coherent_mean']:+.5f} "
              f"2se={2*a['delta_se']:.5f} -> {a['verdict']}")

    # ---------------------------------------------- TASK B: nonlinear purity (S_2)
    for n in sizes_b:
        cells = []
        for sd in seeds:
            rng = np.random.default_rng(20_000 + 97 * sd + n)
            d = 2 ** n
            v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
            psi = v / np.linalg.norm(v)
            rho = _mixed_state(psi, n, depol_p)
            exact = float(np.real(np.trace(rho @ rho)))
            ev = _eigvals_depolarized(n, depol_p)
            signal = abs(1.0 - exact)  # pure (purity 1) vs depolarized

            r_inf = _rmse_informed_B(ev, exact, matched_b, reps_b, rng)
            r_coh = _rmse_coherent_B(exact, matched_b, reps_b, rng)
            r_agn = _rmse_agnostic_B(rho, n, exact, matched_b, reps_b, rng)

            c_inf, b_inf = _copies_to_target(
                lambda M: _rmse_informed_B(ev, exact, M, 60, rng), grid_b_cheap, EPS)
            c_coh, b_coh = _copies_to_target(
                lambda M: _rmse_coherent_B(exact, M, 200, rng), grid_b_cheap, EPS)
            c_agn, b_agn = _copies_to_target(
                lambda M: _rmse_agnostic_B(rho, n, exact, M, 20, rng), grid_b_shadow, EPS)

            cells.append(dict(rmse_informed=r_inf, rmse_coherent=r_coh, rmse_agnostic=r_agn))
            per_seed.append(dict(
                task="B_nonlinear_purity", n=n, seed=sd,
                exact_value=exact, label_signal_scale=signal,
                vacuous_excluded=int(signal < 1e-10),
                matched_copies=matched_b,
                rmse_agnostic=r_agn, rmse_informed=r_inf, rmse_coherent=r_coh,
                delta_informed_minus_coherent=r_inf - r_coh,
                var_per_shot_agnostic="", var_per_shot_informed="",
                var_per_shot_coherent=1.0 - exact ** 2,
                var_agnostic_closed_form="", var_informed_closed_form="",
                copies_to_target_agnostic=c_agn, copies_to_target_agnostic_lower_bound=b_agn,
                copies_to_target_informed=c_inf, copies_to_target_informed_lower_bound=b_inf,
                copies_to_target_coherent=c_coh, copies_to_target_coherent_lower_bound=b_coh,
                shadow_fastpath_check_relerr=""))
        agg.append(_aggregate("B_nonlinear_purity", n, cells, per_seed, matched_b))
        a = agg[-1]
        print(f"[B] n={n}: RMSE agn={a['rmse_agnostic_mean']:.4f} "
              f"inf={a['rmse_informed_mean']:.4f} coh={a['rmse_coherent_mean']:.4f} | "
              f"delta={a['delta_informed_minus_coherent_mean']:+.5f} "
              f"2se={2*a['delta_se']:.5f} -> {a['verdict']}")

    os.makedirs(out_dir, exist_ok=True)
    p1 = os.path.join(out_dir, "informed_comparator.csv")
    p2 = os.path.join(out_dir, "informed_comparator_perseed.csv")
    for path, rows in ((p1, agg), (p2, per_seed)):
        if os.path.exists(path):
            raise SystemExit(f"refusing to overwrite existing results file: {path}")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", path)
    return agg, per_seed


def _aggregate(task, n, cells, per_seed, matched):
    """Paired per-seed deltas: difference first, then average. se = std(ddof=1)/sqrt(n)."""
    rows = [r for r in per_seed if r["task"] == task and r["n"] == n and not r["vacuous_excluded"]]
    d = np.array([r["delta_informed_minus_coherent"] for r in rows], dtype=float)
    m = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    survives = m > 2 * se
    ri = np.array([c["rmse_informed"] for c in cells], dtype=float)
    rc = np.array([c["rmse_coherent"] for c in cells], dtype=float)
    ra = np.array([c["rmse_agnostic"] for c in cells], dtype=float)
    return dict(
        task=task, n=n, n_seeds=len(d), matched_copies=matched,
        label_signal_scale=float(np.mean([r["label_signal_scale"] for r in rows])),
        vacuous_excluded=int(len(d) == 0),
        rmse_agnostic_mean=float(np.mean(ra)), rmse_agnostic_se=float(np.std(ra, ddof=1) / np.sqrt(len(ra))),
        rmse_informed_mean=float(np.mean(ri)), rmse_informed_se=float(np.std(ri, ddof=1) / np.sqrt(len(ri))),
        rmse_coherent_mean=float(np.mean(rc)), rmse_coherent_se=float(np.std(rc, ddof=1) / np.sqrt(len(rc))),
        delta_informed_minus_coherent_mean=m, delta_se=se, delta_two_se=2 * se,
        ratio_agnostic_over_informed=float(np.mean(ra) / np.mean(ri)),
        ratio_informed_over_coherent=float(np.mean(ri) / np.mean(rc)),
        verdict=("SURVIVES" if survives else "KILLED"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes-a", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--sizes-b", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--out", default="results/copy_fusion")
    a = ap.parse_args(argv)
    run(a.sizes_a, a.sizes_b, a.seeds, a.out)


if __name__ == "__main__":
    main()
