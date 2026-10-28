
import numpy as np, itertools

def iqp_circuit(n, seed):
    """A diagonal IQP unitary U = H^n D H^n with random Z/ZZ phases (supremacy class)."""
    rng=np.random.default_rng(seed)
    H=np.array([[1,1],[1,-1]])/np.sqrt(2)
    Hn=np.array([[1.0]])
    for _ in range(n): Hn=np.kron(Hn,H)
    d=1<<n
    # diagonal of exp(i * sum theta_ij Z_i Z_j + phi_i Z_i)
    z=1-2*((np.arange(d)[:,None]>>np.arange(n)[None,:])&1)   # z[s,i] in {+1,-1}
    phase=np.zeros(d)
    for i in range(n):
        phase+=rng.uniform(0,2*np.pi)*z[:,i]
        for j in range(i+1,n):
            phase+=rng.uniform(0,2*np.pi)*z[:,i]*z[:,j]
    D=np.diag(np.exp(1j*phase))
    return Hn@D@Hn

def flip_op(n, k):
    """X on qubit k of an n-qubit register (qubit 0 = most significant)."""
    X=np.array([[0,1],[1,0]]); I=np.eye(2); ops=[I]*n; ops[k]=X
    M=np.array([[1.0]])
    for o in ops: M=np.kron(M,o)
    return M


if __name__ == "__main__":

    print("=== I1: degenerate-clean construction (U_enc lands in H_+) ===")
    # main register carries the IQP circuit; a Z_2 ancilla is prepared in the +1 eigenstate,
    # so the joint state is G-invariant and Pi acts as identity on it.
    for n in (2,3,4):
        U=iqp_circuit(n, 7+n)
        psi=U[:,0]                      # U_enc|0>
        # Pi = identity on this construction (state already invariant); f_A = <psi|M|psi>
        for t in (0, (1<<n)-1):
            M=np.zeros((1<<n,1<<n)); M[t,t]=1
            fA=np.real(psi.conj()@M@psi)
            amp=abs(psi[t])**2
            print(f"  n={n} t={t:>2}: f_A={fA:.6f}  |A(t)|^2={amp:.6f}  err={abs(fA-amp):.2e}")

    print("\n=== I2: orbit-sum for nontrivial Pi (G=Z_2 flipping qubit 0) ===")
    for n in (2,3):
        U=iqp_circuit(n, 11+n); psi=U[:,0]
        X0=flip_op(n,0)
        Pi=(np.eye(1<<n)+X0)/2          # projector onto Z_2-invariant subspace
        Ppsi=Pi@psi
        p=np.real(psi.conj()@Pi@psi)    # acceptance ||Pi psi||^2
        for t in range(min(4,1<<n)):
            M=np.zeros((1<<n,1<<n)); M[t,t]=1
            fA=np.real(Ppsi.conj()@M@Ppsi)/p
            tflip=t^(1<<(n-1))          # t with qubit 0 flipped
            orbit=abs(0.5*(psi[t]+psi[tflip]))**2/p
            print(f"  n={n} t={t:>2}: f_A={fA:.6f}  |A(t)+A(t')|^2/(4p)... orbit={orbit:.6f} err={abs(fA-orbit):.2e}"
                  f"   accept p={p:.4f}")

    # --- write the grounding to a committed CSV ---
    import csv, os
    rows=[]
    for n in (2,3,4,5):
        U=iqp_circuit(n, 7+n); psi=U[:,0]
        for t in (0,(1<<n)-1):
            M=np.zeros((1<<n,1<<n)); M[t,t]=1
            fA=float(np.real(psi.conj()@M@psi)); amp=float(abs(psi[t])**2)
            rows.append(dict(identity="I1_clean", n=n, t=t, f_A=fA, target_quantity=amp,
                             abs_err=abs(fA-amp), accept_p=1.0))
        X0=flip_op(n,0); Pi=(np.eye(1<<n)+X0)/2; Ppsi=Pi@psi
        p=float(np.real(psi.conj()@Pi@psi))
        for t in range(min(4,1<<n)):
            M=np.zeros((1<<n,1<<n)); M[t,t]=1
            fA=float(np.real(Ppsi.conj()@M@Ppsi)/p)
            tf=t^(1<<(n-1)); orb=float(abs(0.5*(psi[t]+psi[tf]))**2/p)
            rows.append(dict(identity="I2_orbit", n=n, t=t, f_A=fA, target_quantity=orb,
                             abs_err=abs(fA-orb), accept_p=p))
    out="results/trifecta/hardness_reduction.csv"
    assert not os.path.exists(out), out
    with open(out,"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows, max abs_err {max(r['abs_err'] for r in rows):.1e})")
