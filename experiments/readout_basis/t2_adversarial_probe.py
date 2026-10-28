
import os, sys, csv
import numpy as np
ROOT = os.path.abspath(__file__)
while ROOT != os.path.dirname(ROOT) and not os.path.exists(
        os.path.join(ROOT, "pyproject.toml")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
from experiments.readout_basis import t2_nonabelian_learned_readout as T2
from cqfa.group import cyclic, dihedral4, symmetric
from scipy.optimize import minimize

OUT = os.path.join(ROOT, "results", "cqfa_pp")
SEEDS = T2.SEEDS


def logreg_diag(Xtr, ytr, Xte, yte, l2, maxiter):
    Xtr, Xte = T2._standardize(Xtr, Xte)
    n, p = Xtr.shape
    yv = ytr.astype(float)
    def obj(th):
        w, b = th[:p], th[p]
        z = Xtr @ w + b
        loss = float(np.mean(np.logaddexp(0., z) - yv*z)) + l2*float(w@w)
        s = 1/(1+np.exp(-z))
        return loss, np.concatenate([Xtr.T@(s-yv)/n + 2*l2*w, [float(np.mean(s-yv))]])
    r = minimize(obj, np.zeros(p+1), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=maxiter, maxfun=10*maxiter))
    w, b = r.x[:p], r.x[p]
    g = np.linalg.norm(obj(r.x)[1], np.inf)
    return dict(acc_tr=float(np.mean(((Xtr@w+b)>0)==ytr)), acc_te=float(np.mean(((Xte@w+b)>0)==yte)),
                nit=int(r.nit), success=int(r.success), gradinf=float(g), fun=float(r.fun))


def knn(Xtr, ytr, Xte, yte, k=15):
    Xtr, Xte = T2._standardize(Xtr, Xte)
    D = ((Xte[:, None, :]-Xtr[None, :, :])**2).sum(-1)
    idx = np.argsort(D, axis=1)[:, :k]
    pred = (ytr[idx].mean(1) > 0.5).astype(int)
    return float(np.mean(pred == yte))


def mlp_strong(Xtr, ytr, Xte, yte, h, seed, restarts, maxiter, l2):
    old_r, old_m, old_l = T2.N_RESTARTS, T2.MAXITER, T2.L2
    T2.N_RESTARTS, T2.MAXITER, T2.L2 = restarts, maxiter, l2
    tr, te, npar = T2.fit_mlp(Xtr, ytr, Xte, yte, h, seed)
    T2.N_RESTARTS, T2.MAXITER, T2.L2 = old_r, old_m, old_l
    return tr, te, npar


rows = []
groups = [("C6", cyclic(6)), ("S3", symmetric(3)), ("C8", cyclic(8)), ("D4", dihedral4())]


if __name__ == "__main__":
    for name, rep in groups:
        m = rep.order
        T = T2.mult_table(rep); L, R = T2.regular_ops(T); ab = T2.is_abelian(T)
        bases = [T2.fourier_basis(L, R, b) for b in range(T2.N_FIXED_BASES)]
        F0 = bases[0]
        # (0) structural: is feat_modeF an exact LINEAR function of feat_full? if yes the learned
        #     hypothesis class strictly CONTAINS the modeF class -> a null cannot be a class deficit.
        rng = np.random.default_rng(3)
        P = rng.normal(size=(50, m)) + 1j*rng.normal(size=(50, m)); P /= np.linalg.norm(P, axis=1, keepdims=True)
        V = np.einsum("gij,nj->ngi", L, P); rh = np.einsum("ngi,nhi->ngh", V, V.conj())/m
        A, B = T2.feat_full(rh), T2.feat_modeF(rh, F0)
        resid = float(np.max(np.abs(A @ np.linalg.lstsq(A, B, rcond=None)[0] - B)))
        for task in ["generic_W", "offdiag_W", "nonlinear_W"]:
            for sd in SEEDS:
                if task == "nonlinear_W":
                    # ANTI-RIGGING: label NOT linear in rho -- XOR of two linear functionals.
                    r_ = np.random.default_rng(1000+sd)
                    n = T2.N_TRAIN + T2.N_TEST
                    Pn = r_.normal(size=(n, m)) + 1j*r_.normal(size=(n, m)); Pn /= np.linalg.norm(Pn, axis=1, keepdims=True)
                    Vn = np.einsum("gij,nj->ngi", L, Pn); rhos = np.einsum("ngi,nhi->ngh", Vn, Vn.conj())/m
                    ys = []
                    for _ in range(2):
                        Hc = r_.normal(size=(m, m)) + 1j*r_.normal(size=(m, m)); W = (Hc+Hc.conj().T)/2
                        v = np.real(np.einsum("nab,ba->n", rhos, W))
                        ys.append((v > np.median(v[:T2.N_TRAIN])).astype(int))
                    y = ys[0] ^ ys[1]
                    Rtr, ytr, Rte, yte = rhos[:T2.N_TRAIN], y[:T2.N_TRAIN], rhos[T2.N_TRAIN:], y[T2.N_TRAIN:]
                    sig = 1.0
                else:
                    Rtr, ytr, Rte, yte, sig = T2.make_dataset(m, L, F0, task, sd)
                Ftr, Fte = T2.feat_modeF(Rtr, F0), T2.feat_modeF(Rte, F0)
                Gtr, Gte = T2.feat_full(Rtr), T2.feat_full(Rte)
                if task == "nonlinear_W":
                    # linear arms cannot express XOR; use the SAME strong MLP on BOTH feature sets
                    a_mf = mlp_strong(Ftr, ytr, Fte, yte, 32, sd, 8, 20000, 1e-3)
                    a_lr = mlp_strong(Gtr, ytr, Gte, yte, 32, sd, 8, 20000, 1e-3)
                    rows.append(dict(group=name, abelian=int(ab), task=task, seed=sd,
                                     arm="mlp32_on_modeF_vs_full", acc_modeF=a_mf[1], acc_learned=a_lr[1],
                                     acc_modeF_train=a_mf[0], acc_learned_train=a_lr[0],
                                     delta=a_lr[1]-a_mf[1], n_params_modeF=a_mf[2], n_params_learned=a_lr[2],
                                     knn_modeF=knn(Ftr, ytr, Fte, yte), knn_full=knn(Gtr, ytr, Gte, yte),
                                     modeF_is_linear_fn_of_full_resid=resid,
                                     modeF_gradinf=np.nan, learned_gradinf=np.nan,
                                     modeF_nit=-1, learned_nit=-1, label_signal_scale=sig))
                    continue
                # A: convergence diagnostics on the two PRIMARY arms
                dmf = logreg_diag(Ftr, ytr, Fte, yte, T2.L2, 200000)
                dlr = logreg_diag(Gtr, ytr, Gte, yte, T2.L2, 200000)
                # B: MUCH stronger fixed-basis arm -- big MLP, 8 restarts, 20k iters, + kNN (nonparametric)
                s_tr, s_te, s_np = mlp_strong(Ftr, ytr, Fte, yte, 32, sd, 8, 20000, 1e-3)
                rows.append(dict(group=name, abelian=int(ab), task=task, seed=sd,
                                 arm="convex+strongcontrol",
                                 acc_modeF=dmf["acc_te"], acc_learned=dlr["acc_te"],
                                 acc_modeF_train=dmf["acc_tr"], acc_learned_train=dlr["acc_tr"],
                                 delta=dlr["acc_te"]-dmf["acc_te"],
                                 modeF_gradinf=dmf["gradinf"], learned_gradinf=dlr["gradinf"],
                                 modeF_nit=dmf["nit"], learned_nit=dlr["nit"],
                                 acc_modeF_strongMLP=s_te, acc_modeF_strongMLP_train=s_tr,
                                 n_params_modeF_strongMLP=s_np,
                                 knn_modeF=knn(Ftr, ytr, Fte, yte), knn_full=knn(Gtr, ytr, Gte, yte),
                                 modeF_is_linear_fn_of_full_resid=resid,
                                 n_params_modeF=len(Ftr[0])+1, n_params_learned=len(Gtr[0])+1,
                                 label_signal_scale=sig))
            sub = rows[-len(SEEDS):]
            d = np.array([x["delta"] for x in sub]); se = float(np.std(d, ddof=1)/np.sqrt(len(d)))
            print(f"{name:>3} ab={int(ab)} {task:>12} modeF={np.mean([x['acc_modeF'] for x in sub]):.4f} "
                  f"strongMLP={np.mean([x.get('acc_modeF_strongMLP', np.nan) for x in sub]):.4f} "
                  f"knnF={np.mean([x['knn_modeF'] for x in sub]):.4f} knnFull={np.mean([x['knn_full'] for x in sub]):.4f} "
                  f"learned={np.mean([x['acc_learned'] for x in sub]):.4f} "
                  f"trainF={np.mean([x['acc_modeF_train'] for x in sub]):.4f} trainL={np.mean([x['acc_learned_train'] for x in sub]):.4f} "
                  f"| d={d.mean():+.4f} 2se={2*se:.4f} -> {'SURVIVES' if d.mean()>2*se else 'negative'} "
                  f"| resid(modeF in span full)={resid:.2e}", flush=True)

    cols = sorted({k for r in rows for k in r})
    p = os.path.join(OUT, "t2_adversarial_probe.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print("wrote", p)
