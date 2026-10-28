
import os
import sys
import csv

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from cqfa.paths import RESULTS, ensure

_ROOT = os.path.abspath(__file__)
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(
        os.path.join(_ROOT, "pyproject.toml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from experiments.readout_basis.t1_abelian_convergence import (dft, orbit_matrix, feats, herm, _unpack, nll,
                                    centroid_acc, d_to_dft, train_readout)
from experiments.fisher_information.fourier_mode import hidden_shift_dataset

OUT = str(ensure(RESULTS / "readout_basis"))
SEEDS = [0, 1, 2, 3, 4]


def logm_herm(U):
    """theta such that expm(i herm(theta)) == U, via eigendecomposition."""
    w, V = np.linalg.eig(U)
    H = (V * np.angle(w)) @ np.linalg.inv(V)
    H = (H + H.conj().T) / 2
    N = U.shape[0]
    iu = np.triu_indices(N, 1)
    return np.concatenate([H[np.diag_indices(N)].real, H[iu].real, H[iu].imag])


def head_only_loss(R, N, Wtr, Wref, ytr, reg, shots, Ztr, maxiter=4000):
    """Best achievable train loss with R FROZEN (head trained). Loss at the point R."""
    th = logm_herm(R)
    rng = np.random.default_rng(5)
    best = np.inf
    for _ in range(3):
        x0h = np.concatenate([0.1 * rng.standard_normal(3 * N), [0.0]])
        f = lambda h: nll(np.concatenate([th, h]), N, Wtr, Wref, ytr, reg, shots, Ztr)
        r = minimize(f, x0h, method="L-BFGS-B", options=dict(maxiter=maxiter, maxfun=200000))
        best = min(best, float(r.fun))
    return best, th


def warm_start_at_F(N, Wtr, Wref, ytr, reg, shots, Ztr, maxiter=4000):
    """Full end-to-end training STARTING at R = DFT."""
    _, thF = head_only_loss(dft(N), N, Wtr, Wref, ytr, reg, shots, Ztr)
    rng = np.random.default_rng(7)
    x0 = np.concatenate([thF, 0.1 * rng.standard_normal(3 * N), [0.0]])
    r = minimize(nll, x0, args=(N, Wtr, Wref, ytr, reg, shots, Ztr),
                 method="L-BFGS-B", options=dict(maxiter=maxiter, maxfun=200000))
    th, _, _ = _unpack(r.x, N)
    return expm(1j * herm(th, N)), float(r.fun)


def setup(N, seed, m_train, m_test):
    rng = np.random.default_rng(seed)
    phi, data = hidden_shift_dataset(N, m_train + m_test, rng)
    Wall = np.stack([orbit_matrix(d[0], N) for d in data])
    Wref = orbit_matrix(phi, N)
    y = np.array([d[2] for d in data])
    nrng = np.random.default_rng(777_000 + 97 * seed + N)
    Zall = tuple(nrng.standard_normal((m_train + m_test, N)) for _ in range(3))
    return (Wall[:m_train], Wall[m_train:], Wref, y[:m_train], y[m_train:],
            tuple(z[:m_train] for z in Zall), tuple(z[m_train:] for z in Zall))


rows = []


if __name__ == "__main__":

    print("=" * 78)
    print("ATTACK A+B: under-training test + optimiser stress")
    print("=" * 78)
    for (N, shots, m_train) in [(8, None, 60), (8, 16, 60), (16, 16, 60), (16, None, 60)]:
        for sd in SEEDS:
            Wtr, Wte, Wref, ytr, yte, Ztr, Zte = setup(N, sd, m_train, 40)
            reg = 1e-4
            lossF, _ = head_only_loss(dft(N), N, Wtr, Wref, ytr, reg, shots, Ztr)
            # weak (as-shipped) optimiser
            Rw, lw = train_readout(N, Wtr, Wref, ytr, sd, 3, 400, reg, shots, Ztr)
            # stressed optimiser
            Rs, ls = train_readout(N, Wtr, Wref, ytr, sd, 12, 4000, reg, shots, Ztr)
            # warm start at F
            Rf, lf = warm_start_at_F(N, Wtr, Wref, ytr, reg, shots, Ztr)
            F = dft(N)
            aF = centroid_acc(feats(F, Wtr, Wref, shots, Ztr), ytr, feats(F, Wte, Wref, shots, Zte), yte)
            aw = centroid_acc(feats(Rw, Wtr, Wref, shots, Ztr), ytr, feats(Rw, Wte, Wref, shots, Zte), yte)
            as_ = centroid_acc(feats(Rs, Wtr, Wref, shots, Ztr), ytr, feats(Rs, Wte, Wref, shots, Zte), yte)
            af = centroid_acc(feats(Rf, Wtr, Wref, shots, Ztr), ytr, feats(Rf, Wte, Wref, shots, Zte), yte)
            rows.append(dict(attack="A_undertrain", N=N, shots=("exact" if shots is None else shots),
                             m_train=m_train, seed=sd,
                             loss_at_F=round(lossF, 6), loss_weak=round(lw, 6),
                             loss_stress=round(ls, 6), loss_warmF=round(lf, 6),
                             acc_modeF=round(aF, 6), acc_weak=round(aw, 6),
                             acc_stress=round(as_, 6), acc_warmF=round(af, 6),
                             d_weak=round(d_to_dft(Rw, N), 6), d_stress=round(d_to_dft(Rs, N), 6),
                             d_warmF=round(d_to_dft(Rf, N), 6)))
            print(f"  N={N} shots={str(shots):>5} sd={sd}: loss@F={lossF:.5f} weak={lw:.5f} "
                  f"stress={ls:.5f} warmF={lf:.5f} | accF={aF:.3f} weak={aw:.3f} "
                  f"stress={as_:.3f} warmF={af:.3f} | d_warmF={d_to_dft(Rf,N):.4f}")
        r = [x for x in rows if x["N"] == N and x["shots"] == ("exact" if shots is None else shots)]
        g = lambda k: np.array([x[k] for x in r])
        print(f"N={N} shots={shots} MEAN over {len(r)} seeds: loss@F={g('loss_at_F').mean():.5f} "
              f"weak={g('loss_weak').mean():.5f} stress={g('loss_stress').mean():.5f} | "
              f"delta_weak={(g('acc_weak')-g('acc_modeF')).mean():+.3f} "
              f"delta_stress={(g('acc_stress')-g('acc_modeF')).mean():+.3f}")
        print()

    print("=" * 78)
    print("ATTACK C: power check -- shrink m_train to escape saturation (exact budget)")
    print("=" * 78)
    for N in [8, 16]:
        for m_train in [8, 16, 30]:
            aFs, aLs, ds = [], [], []
            for sd in SEEDS:
                Wtr, Wte, Wref, ytr, yte, Ztr, Zte = setup(N, sd, m_train, 40)
                F = dft(N)
                aF = centroid_acc(feats(F, Wtr, Wref, None, Ztr), ytr, feats(F, Wte, Wref, None, Zte), yte)
                R, _ = train_readout(N, Wtr, Wref, ytr, sd, 12, 4000, 1e-4, None, Ztr)
                aL = centroid_acc(feats(R, Wtr, Wref, None, Ztr), ytr, feats(R, Wte, Wref, None, Zte), yte)
                aFs.append(aF); aLs.append(aL); ds.append(d_to_dft(R, N))
                rows.append(dict(attack="C_power", N=N, shots="exact", m_train=m_train, seed=sd,
                                 loss_at_F="", loss_weak="", loss_stress="", loss_warmF="",
                                 acc_modeF=round(aF, 6), acc_weak="", acc_stress=round(aL, 6),
                                 acc_warmF="", d_weak="", d_stress=round(ds[-1], 6), d_warmF=""))
            aFs, aLs, ds = map(np.array, (aFs, aLs, ds))
            print(f"  N={N} m_train={m_train:>2}: modeF={aFs.mean():.3f}+-{aFs.std():.3f} "
                  f"learned={aLs.mean():.3f}+-{aLs.std():.3f} delta={(aLs-aFs).mean():+.3f}"
                  f"+-{(aLs-aFs).std():.3f}  d={ds.mean():.4f}")

    p = os.path.join(OUT, "t1_verify_probe.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\nwrote", p)
