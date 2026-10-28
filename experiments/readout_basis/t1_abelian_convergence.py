
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize, linear_sum_assignment

_ROOT = os.path.abspath(__file__)
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(
        os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from experiments.fisher_information.fourier_mode import (  # noqa: E402
    shift, mode_F_components, hidden_shift_dataset)


# ---------------------------------------------------------------------------
# readout family
# ---------------------------------------------------------------------------
def dft(N: int) -> np.ndarray:
    
    k = np.arange(N)[:, None]
    g = np.arange(N)[None, :]
    return np.exp(2j * np.pi * k * g / N) / np.sqrt(N)


def orbit_matrix(psi: np.ndarray, N: int) -> np.ndarray:
    """rows g = U_g psi, scaled by 1/sqrt(N)  (the state after the controlled oracle)."""
    return np.stack([shift(N, g) @ psi for g in range(N)]) / np.sqrt(N)


def components(R: np.ndarray, W: np.ndarray) -> np.ndarray:
    """v_j = sum_g R_{jg} (U_g psi)/sqrt(N).  W may be (N,N) or (m,N,N)."""
    return R @ W


def feats(R: np.ndarray, Wall: np.ndarray, Wref: np.ndarray, shots=None, Z=None) -> np.ndarray:
    
    V = components(R, Wall)                       # (m, N, dim)
    V0 = components(R, Wref)                      # (N, dim)
    p = np.sum(np.abs(V) ** 2, axis=2)            # (m, N)
    z = np.einsum("jd,mjd->mj", V0.conj(), V)     # (m, N)
    a = np.abs(z) + 1e-15
    c, s = z.real / a, z.imag / a
    if shots is not None:
        Zp, Zc, Zs = Z
        S = float(shots)
        p = p + Zp * np.sqrt(np.clip(p * (1.0 - p), 0.0, None) / S)
        p = np.clip(p, 0.0, 1.0)
        n = np.clip(S * p, 1.0, None)
        c = c + Zc * np.sqrt(np.clip(1.0 - c ** 2, 1e-6, None) / n)
        s = s + Zs * np.sqrt(np.clip(1.0 - s ** 2, 1e-6, None) / n)
        c, s = np.clip(c, -1.0, 1.0), np.clip(s, -1.0, 1.0)
    return np.concatenate([p, c, s], axis=1)


# ---------------------------------------------------------------------------
# sanity: R = DFT reproduces mode_F_components exactly
# ---------------------------------------------------------------------------
def _selfcheck(N=6, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    psi = v / np.linalg.norm(v)
    ours = components(dft(N), orbit_matrix(psi, N))
    theirs, _ = mode_F_components(psi, N)
    err = max(np.max(np.abs(ours[k] - theirs[k])) for k in range(N))
    assert err < 1e-12, f"readout family does not contain Mode F (err={err})"
    return err


# ---------------------------------------------------------------------------
# Hermitian parameterisation and training
# ---------------------------------------------------------------------------
def herm(theta: np.ndarray, N: int) -> np.ndarray:
    iu = np.triu_indices(N, 1)
    H = np.zeros((N, N), dtype=complex)
    H[np.diag_indices(N)] = theta[:N]
    nod = len(iu[0])
    H[iu] = theta[N:N + nod] + 1j * theta[N + nod:N + 2 * nod]
    return H + H.conj().T - np.diag(np.diag(H).real)


def _unpack(x, N):
    nH = N * N
    return x[:nH], x[nH:nH + 3 * N], x[nH + 3 * N]


def nll(x, N, Wtr, Wref, ytr, reg, shots, Ztr):
    th, w, b = _unpack(x, N)
    R = expm(1j * herm(th, N))
    X = feats(R, Wtr, Wref, shots, Ztr)
    z = X @ w + b
    # stable logistic loss
    loss = np.mean(np.logaddexp(0.0, z) - ytr * z)
    return loss + reg * float(w @ w)


def train_readout(N, Wtr, Wref, ytr, seed, n_init, maxiter, reg, shots, Ztr):
    """Train H and logistic head end-to-end; keep the init with the best TRAIN loss."""
    rng = np.random.default_rng(10_000 + seed)
    nH = N * N
    best = None
    for _ in range(n_init):
        x0 = np.concatenate([0.6 * rng.standard_normal(nH),
                             0.1 * rng.standard_normal(3 * N), [0.0]])
        res = minimize(nll, x0, args=(N, Wtr, Wref, ytr, reg, shots, Ztr),
                       method="L-BFGS-B", options=dict(maxiter=maxiter, maxfun=200000))
        if best is None or res.fun < best.fun:
            best = res
    th, _, _ = _unpack(best.x, N)
    return expm(1j * herm(th, N)), float(best.fun)


# ---------------------------------------------------------------------------
# evaluation: nearest-centroid trained on train split, scored on test split
# ---------------------------------------------------------------------------
def centroid_acc(Xtr, ytr, Xte, yte):
    Xtr = np.asarray(Xtr, float); Xte = np.asarray(Xte, float)
    sd = Xtr.std(0)
    keep = sd > 1e-8 * (np.abs(Xtr).mean() + 1e-12)
    maj = max(float(np.mean(yte)), 1.0 - float(np.mean(yte)))
    if not np.any(keep):
        return maj
    mu = Xtr[:, keep].mean(0)
    A = (Xtr[:, keep] - mu) / sd[keep]
    B = (Xte[:, keep] - mu) / sd[keep]
    c0 = A[ytr == 0].mean(0) if np.any(ytr == 0) else np.zeros(A.shape[1])
    c1 = A[ytr == 1].mean(0) if np.any(ytr == 1) else np.zeros(A.shape[1])
    pred = (np.linalg.norm(B - c1, axis=1) < np.linalg.norm(B - c0, axis=1)).astype(int)
    return float(np.mean(pred == yte))


def d_to_dft(R: np.ndarray, N: int) -> float:
    """1 - max over diagonal phases D and permutations P of (1/N)|Tr(R^dag F D P)|."""
    C = np.abs(R.conj() @ dft(N).T)          # C[j,k] = |<R_j, F_k>|
    r, c = linear_sum_assignment(-C)
    return float(1.0 - C[r, c].sum() / N)


# ---------------------------------------------------------------------------
def run(sizes, m_train, m_test, seeds, n_init, maxiter, reg, shot_grid, out):
    err = _selfcheck()
    print(f"[selfcheck] R=DFT reproduces mode_F_components, max err = {err:.2e}")
    rows = []
    for N in sizes:
        for shots in shot_grid:
            t0 = time.time()
            sub = []
            for sd in seeds:
                rng = np.random.default_rng(sd)
                phi, data = hidden_shift_dataset(N, m_train + m_test, rng)
                Wall = np.stack([orbit_matrix(d[0], N) for d in data])
                Wref = orbit_matrix(phi, N)
                y = np.array([d[2] for d in data])
                Wtr, Wte = Wall[:m_train], Wall[m_train:]
                ytr, yte = y[:m_train], y[m_train:]

                # fixed measurement-noise draws, SHARED by Mode F and the learned readout
                nrng = np.random.default_rng(777_000 + 97 * sd + N)
                Zall = tuple(nrng.standard_normal((m_train + m_test, N)) for _ in range(3))
                Ztr = tuple(z[:m_train] for z in Zall)
                Zte = tuple(z[m_train:] for z in Zall)

                F = dft(N)
                acc_F = centroid_acc(feats(F, Wtr, Wref, shots, Ztr), ytr,
                                     feats(F, Wte, Wref, shots, Zte), yte)
                R, trloss = train_readout(N, Wtr, Wref, ytr, sd, n_init, maxiter, reg,
                                          shots, Ztr)
                acc_L = centroid_acc(feats(R, Wtr, Wref, shots, Ztr), ytr,
                                     feats(R, Wte, Wref, shots, Zte), yte)

               
                pF_tr = feats(F, Wtr, Wref, shots, Ztr)[:, :N]
                pF_te = feats(F, Wte, Wref, shots, Zte)[:, :N]
                pL_tr = feats(R, Wtr, Wref, shots, Ztr)[:, :N]
                pL_te = feats(R, Wte, Wref, shots, Zte)[:, :N]
                acc_Fp = centroid_acc(pF_tr, ytr, pF_te, yte)
                acc_Lp = centroid_acc(pL_tr, ytr, pL_te, yte)
                # exact shift-(in)variance of the populations, no shot noise
                var_F = float(np.mean(np.var(feats(F, Wall, Wref)[:, :N], axis=0)))
                var_L = float(np.mean(np.var(feats(R, Wall, Wref)[:, :N], axis=0)))

                sub.append(dict(N=N, seed=sd, acc_modeF=round(acc_F, 6),
                                acc_learned=round(acc_L, 6),
                                d_to_DFT=round(d_to_dft(R, N), 6),
                                budget=(f"shots={'exact' if shots is None else shots};"
                                        f"m_train={m_train};m_test={m_test}"),
                                shots=("exact" if shots is None else shots),
                                m_train=m_train, m_test=m_test,
                                n_params=N * N + 3 * N + 1, n_init=n_init,
                                train_loss=round(trloss, 6),
                                acc_modeF_probs_only=round(acc_Fp, 6),
                                acc_learned_probs_only=round(acc_Lp, 6),
                                pop_shift_var_modeF=float(f"{var_F:.6e}"),
                                pop_shift_var_learned=float(f"{var_L:.6e}")))
                print(f"  N={N:>3} shots={str(shots):>5} seed={sd}: "
                      f"acc_modeF={acc_F:.3f}  acc_learned={acc_L:.3f}  "
                      f"d_to_DFT={sub[-1]['d_to_DFT']:.4f}  trainloss={trloss:.4f}")
            rows.extend(sub)
            aF = np.array([r["acc_modeF"] for r in sub])
            aL = np.array([r["acc_learned"] for r in sub])
            dd = np.array([r["d_to_DFT"] for r in sub])
            dlt = aL - aF
            print(f"N={N:>3} shots={str(shots):>5} SUMMARY over {len(aF)} seeds: "
                  f"modeF={aF.mean():.3f}+-{aF.std():.3f} learned={aL.mean():.3f}+-{aL.std():.3f} "
                  f"delta={dlt.mean():+.3f}+-{dlt.std():.3f}  "
                  f"d_to_DFT={dd.mean():.4f}+-{dd.std():.4f}   [{time.time()-t0:.1f}s]")
            pf = np.array([r["acc_modeF_probs_only"] for r in sub])
            pl = np.array([r["acc_learned_probs_only"] for r in sub])
            vf = np.array([r["pop_shift_var_modeF"] for r in sub])
            vl = np.array([r["pop_shift_var_learned"] for r in sub])
            print(f"        DIAG probs-only acc: modeF={pf.mean():.3f} learned={pl.mean():.3f} "
                  f"| exact population shift-variance: modeF={vf.mean():.2e} learned={vl.mean():.2e}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", out)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 6, 8, 16])
    ap.add_argument("--m-train", type=int, default=60)
    ap.add_argument("--m-test", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-init", type=int, default=3)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--reg", type=float, default=1e-4)
    # PRE-REGISTERED shot grid (fixed before looking at any result); "exact" = statevector
    ap.add_argument("--shots", nargs="+", default=["16", "64", "256", "exact"])
    ap.add_argument("--out", default=os.path.join(_ROOT, "results/readout_basis/t1_abelian_convergence.csv"))
    a = ap.parse_args(argv)
    grid = [None if s == "exact" else int(s) for s in a.shots]
    run(a.sizes, a.m_train, a.m_test, a.seeds, a.n_init, a.maxiter, a.reg, grid, a.out)


if __name__ == "__main__":
    main()
