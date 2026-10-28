
import csv, os
import numpy as np

def shift(N, g):
    return np.roll(np.eye(N, dtype=complex), g, axis=0)

def rho_frame(psi, N):
    return np.array([[psi.conj() @ (shift(N, h).conj().T @ shift(N, g)) @ psi
                      for h in range(N)] for g in range(N)]) / N

def dft(N):
    return np.array([[np.exp(-2j*np.pi*a*b/N) for b in range(N)]
                     for a in range(N)]) / np.sqrt(N)

def main():
    rows = []
    for seed in range(5):                      # >=5 seeds
        rng = np.random.default_rng(seed)
        for N in [3, 4, 5, 6, 8]:
            cerr = derr = 0.0
            for _ in range(20):                # 20 random states per (seed,N)
                v = rng.standard_normal(N) + 1j*rng.standard_normal(N)
                psi = v / np.linalg.norm(v)
                R = rho_frame(psi, N)
                c = np.array([R[k % N, 0] for k in range(N)])
                C = np.array([[c[(g-h) % N] for h in range(N)] for g in range(N)])
                cerr = max(cerr, float(np.abs(R - C).max()))
                F = dft(N)
                D = F @ R @ F.conj().T
                derr = max(derr, float(np.abs(D - np.diag(np.diag(D))).max()))
            rows.append(dict(seed=seed, N=N, n_states=20,
                             max_abs_dev_from_circulant=cerr,
                             max_abs_offdiag_after_DFT=derr))
    out = os.path.join("results", "readout_dial", "frame_circulant_check.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", out)
    print("global max dev from circulant :", max(r["max_abs_dev_from_circulant"] for r in rows))
    print("global max off-diag after DFT :", max(r["max_abs_offdiag_after_DFT"] for r in rows))

if __name__ == "__main__":
    main()
