
from __future__ import annotations

import sys

import numpy as np

import cqfa
from cqfa.paths import RESULTS, ROOT


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    return ok


def main() -> int:
    print(f"CQFA smoke test\n  archive root: {ROOT}\n")
    results = []

    rep = cqfa.group.z2_rc()
    results.append(check("group construction", rep.order == 2 and rep.dim == 4,
                         f"|G|={rep.order} dim={rep.dim} frame qubits={rep.n_frame_qubits}"))

    P = rep.projector()
    results.append(check("projector is idempotent",
                         np.abs(P @ P - P).max() < 1e-12,
                         f"max |P^2 - P| = {np.abs(P @ P - P).max():.2e}"))

    V = rep.controlled_oracle()
    err = np.abs(V.conj().T @ V - np.eye(V.shape[0])).max()
    results.append(check("group oracle is unitary", err < 1e-12, f"max error = {err:.2e}"))

    rng = np.random.default_rng(0)
    psi = cqfa.tasks.random_states(rep.dim, 1, rng)[0]
    M = cqfa.tasks.random_hermitian(rep.dim, np.random.default_rng(1))
    f_a, p = cqfa.frame.cqfa_readout(psi, rep, M)
    f_b = cqfa.frame.classical_fa_readout(psi, rep, M)
    results.append(check("Mode A and Mode B evaluate",
                         f_a is not None and np.isfinite(f_b),
                         f"f_A={f_a:+.4f} (acceptance {p:.3f})  f_B={f_b:+.4f}"))

    worst = max(abs(cqfa.frame.cqfa_readout(U @ psi, rep, M)[0] - f_a)
                for U in rep.matrices)
    results.append(check("scalar readouts are exactly G-invariant", worst < 1e-12,
                         f"max |f_A(U psi) - f_A(psi)| = {worst:.2e}"))

    for name, builder in (("Z2-rc", cqfa.group.z2_rc),
                          ("Z2-pauli4", lambda: cqfa.group.z2_pauli(4))):
        out = cqfa.verify.verify_equivariance(builder())
        ok = out["err_fa"] < 1e-12 and out["err_cqfa"] < 1e-12 and out["err_base"] > 1e-3
        results.append(check(f"invariance identities hold for {name}", ok,
                             f"Mode A {out['err_cqfa']:.1e}, Mode B {out['err_fa']:.1e}, "
                             f"unsymmetrised control {out['err_base']:.2f}"))

    n_csv = len(list(RESULTS.rglob("*.csv")))
    results.append(check("recorded results are present", n_csv > 100,
                         f"{n_csv} CSV files under results/"))

    curated = RESULTS / "curated"
    results.append(check("curated results are present", curated.is_dir(),
                         f"{len(list(curated.rglob('*.csv')))} CSV files under results/curated/"))

    passed = sum(results)
    print(f"\n  {passed} of {len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
