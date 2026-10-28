
import numpy as np
import itertools, csv, os

def clean_oracle(phi_vals, m, a):

    dim = 2 ** (m + a + 1)
    U = np.zeros((dim, dim))
    for idx in range(dim):
        x = idx >> (a + 1)
        anc = (idx >> 1) & ((1 << a) - 1)
        y = idx & 1
        # phi is applied to the output qubit only in the clean subspace anc==0
        if anc == 0:
            y_new = y ^ phi_vals[x]
        else:
            y_new = y
        jdx = (x << (a + 1)) | (anc << 1) | y_new
        U[jdx, idx] = 1.0
    return U

def had_m(m, a):
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    Hm = np.array([[1.0]])
    for _ in range(m):
        Hm = np.kron(Hm, H)
    return np.kron(Hm, np.eye(2 ** (a + 1)))

rows = []
rng = np.random.default_rng(0)


if __name__ == "__main__":
    for m in range(1, 6):
        a = 1  # one spare ancilla; result is independent of a
        # random boolean function phi on m bits
        phi_vals = rng.integers(0, 2, size=2 ** m)
        N_phi = int(phi_vals.sum())
        Hm = had_m(m, a)
        V = clean_oracle(phi_vals, m, a)
        U = Hm @ V @ Hm
        dim = 2 ** (m + a + 1)
        zero = np.zeros(dim); zero[0] = 1.0          # |0^m 0^a 0>
        t = (0 << (a + 1)) | (0 << 1) | 1            # |0^m 0^a 1>
        psi = U @ zero
        A_t = psi[t]
        f_A = abs(A_t) ** 2
        N_rec = 2 ** m * np.sqrt(f_A)
        rows.append(dict(
            m=m, N_phi=N_phi,
            A_t_real=float(A_t.real), A_t_imag=float(A_t.imag),
            pred_A=2 ** (-m) * N_phi,
            f_A=float(f_A), N_recovered=float(N_rec),
            recover_err=abs(N_rec - N_phi),
        ))

    out = os.path.join(os.path.dirname(__file__),
                       "..", "..", "results", "e11_trifecta",
                       "sat_reduction_gadget.csv")
    out = os.path.abspath(out)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    max_imag = max(abs(r["A_t_imag"]) for r in rows)
    max_Aerr = max(abs(r["A_t_real"] - r["pred_A"]) for r in rows)
    max_rec = max(r["recover_err"] for r in rows)
    print(f"wrote {out}")
    print(f"max |Im A(t)|            = {max_imag:.2e}")
    print(f"max |A(t) - 2^-m N_phi|  = {max_Aerr:.2e}")
    print(f"max |N_recovered - N_phi|= {max_rec:.2e}")
    for r in rows:
        print(f"  m={r['m']} N_phi={r['N_phi']:2d} A(t)={r['A_t_real']:.6f} "
              f"f_A={r['f_A']:.6e} N_rec={r['N_recovered']:.10f}")
