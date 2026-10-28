
from __future__ import annotations

import csv
import os
import time

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate, StatePreparation, QFTGate
from qiskit.quantum_info import Operator, Statevector

from cqfa.group import (
    Representation, close_matrices, kron, _perm_matrix,
    z2_rc, cyclic, symmetric, dihedral4, X, Z,
)
from cqfa.frame import coherent_projection


BASIS_HERON = ["rz", "sx", "x", "cz"]        # IBM Heron native basis
BASIS_TEXTBOOK = ["u", "cx"]                 # literature-comparable basis
OPT_LEVEL = 3
N_TRANSPILE_SEEDS = 5
TRANSPILE_SEEDS = [11, 22, 33, 44, 55]
EPS_TARGET = 0.01                            # additive precision target
QSD_TOTAL_QUBIT_CAP = 6                      # monolithic-unitary synthesis cap
LINEAR_COUPLING_GATE_CAP = 30000             # routing cap (see integrity rule 7)

OUT_DIR = os.path.join("results", "multitask_benchmark")



def quaternion8() -> Representation:
    """Q8 = {+-1, +-i, +-j, +-k}, faithful 2-dimensional complex irrep on C^2.

    i -> [[i,0],[0,-i]],  j -> [[0,1],[-1,0]].  Non-abelian, order 8, and its
    only faithful irrep is 2-dimensional -- the sharpest contrast with the
    abelian groups, whose irreps are all 1-dimensional.
    """
    gi = np.array([[1j, 0], [0, -1j]], dtype=complex)
    gj = np.array([[0, 1], [-1, 0]], dtype=complex)
    return Representation("Q8", close_matrices([gi, gj], 2), 2)


def alternating4() -> Representation:
    """A4 (order 12), even permutations of 4 points, natural rep on C^4."""
    a = _perm_matrix([1, 2, 0, 3])          # 3-cycle (0 1 2)
    b = _perm_matrix([1, 0, 3, 2])          # double transposition (0 1)(2 3)
    return Representation("A4", close_matrices([a, b], 4), 4)


def cyclic_ordered(n: int) -> Representation:

    P = _perm_matrix([(i + 1) % n for i in range(n)])
    mats, cur = [], np.eye(n, dtype=complex)
    for _ in range(n):
        mats.append(cur.copy())
        cur = cur @ P
    return Representation(f"C{n}", mats, n)


def cyclic_diag8_on_C4() -> Representation:
  
    w = np.exp(2j * np.pi / 8)
    P = np.diag([1.0, w, w ** 2, w ** 3]).astype(complex)
    mats, cur = [], np.eye(4, dtype=complex)
    for _ in range(8):
        mats.append(cur.copy())
        cur = cur @ P
    return Representation("C8-diag-on-C4", mats, 4)


def z2_ordered() -> Representation:
    """Z2 (reverse-complement flavour) with elements ordered [I, A], A = X (x) Z."""
    A = kron([X, Z])
    return Representation("Z2-rc", [np.eye(4, dtype=complex), A], 4)


GROUPS = [
    ("Z2", z2_ordered),
    ("C4", lambda: cyclic_ordered(4)),
    ("C8", lambda: cyclic_ordered(8)),
    ("C8d", cyclic_diag8_on_C4),      # matched abelian control for D4
    ("S3", lambda: symmetric(3)),
    ("D4", dihedral4),
    ("Q8", quaternion8),
    ("A4", alternating4),
    ("S4", lambda: symmetric(4)),
]


# ---------------------------------------------------------------------------
# padding: qubit registers hold 2**m amplitudes, reps may not
# ---------------------------------------------------------------------------
def pad_to_qubits(U: np.ndarray) -> np.ndarray:
    """Embed a d x d unitary into 2**ceil(log2 d) dims as U (+) I."""
    d = U.shape[0]
    m = max(1, int(np.ceil(np.log2(d))))
    D = 2 ** m
    if D == d:
        return U.astype(complex)
    out = np.eye(D, dtype=complex)
    out[:d, :d] = U
    return out


def padded_matrices(rep: Representation):
    return [pad_to_qubits(U) for U in rep.matrices]


def n_data_qubits(rep: Representation) -> int:
    return max(1, int(np.ceil(np.log2(rep.dim))))


# ---------------------------------------------------------------------------
# group structure: abelian test, multiplication table, regular reps, irreps
# ---------------------------------------------------------------------------
def is_abelian(mats, tol=1e-9) -> bool:
    for A in mats:
        for B in mats:
            if not np.allclose(A @ B, B @ A, atol=tol):
                return False
    return True


def multiplication_table(mats, tol=1e-8) -> np.ndarray:
    n = len(mats)
    tab = np.full((n, n), -1, dtype=int)
    for a in range(n):
        for b in range(n):
            P = mats[a] @ mats[b]
            for c in range(n):
                if np.allclose(P, mats[c], atol=tol):
                    tab[a, b] = c
                    break
            if tab[a, b] < 0:
                raise ValueError("representation not closed under multiplication")
    return tab


def regular_reps(tab: np.ndarray):
  
    n = tab.shape[0]
    inv = np.array([int(np.where(tab[a] == 0)[0][0]) for a in range(n)])
    L, R = [], []
    for a in range(n):
        Lm = np.zeros((n, n), dtype=complex)
        Rm = np.zeros((n, n), dtype=complex)
        for b in range(n):
            Lm[tab[a, b], b] = 1.0
            Rm[tab[b, inv[a]], b] = 1.0
        L.append(Lm)
        R.append(Rm)
    return L, R, inv


def group_fourier_transform(mats, abelian: bool, seed: int = 7):
   
    n = len(mats)
    tab = multiplication_table(mats)
    L, R, _ = regular_reps(tab)

    def check(F, block_dims):
        """Unitary + block-diagonalising + every block irreducible (Schur)."""
        if not np.allclose(F @ F.conj().T, np.eye(n), atol=1e-8):
            return False
        if sum(block_dims) != n:
            return False
        offsets = np.cumsum([0] + list(block_dims))
        mask = np.ones((n, n), dtype=bool)
        for i, d in enumerate(block_dims):
            s = offsets[i]
            mask[s:s + d, s:s + d] = False
        chars = np.zeros((len(block_dims), n), dtype=complex)
        for a in range(n):
            B = F @ L[a] @ F.conj().T
            if np.abs(B[mask]).max() > 1e-7:
                return False
            for i, d in enumerate(block_dims):
                s = offsets[i]
                chars[i, a] = np.trace(B[s:s + d, s:s + d])
        for i in range(len(block_dims)):
            schur = float(np.real(np.sum(np.abs(chars[i]) ** 2)) / n)
            if abs(schur - 1.0) > 1e-6:      # >1 means the block is reducible
                return False
        return True

    if abelian:
        w = np.exp(-2j * np.pi / n)
        F = np.array([[w ** (k * g) for g in range(n)] for k in range(n)]) / np.sqrt(n)
        block_dims = [1] * n
        assert check(F, block_dims), "abelian character DFT failed verification"
        return F, block_dims

    
    for trial in range(40):
        rng = np.random.default_rng(seed + 1000 * trial)
        c = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        M = sum(c[a] * R[a] for a in range(n))
        M = (M + M.conj().T) / 2.0
        evals, V = np.linalg.eigh(M)
        scale = max(1.0, float(np.max(np.abs(evals))))
        cut = 1e-8 * scale
        block_dims, run = [], 1
        for gp in np.diff(evals):
            if gp > cut:
                block_dims.append(run)
                run = 1
            else:
                run += 1
        block_dims.append(run)
        F = V.conj().T
        if check(F, block_dims):
            return F, block_dims
    raise RuntimeError("could not construct a verified group Fourier transform")


def irrep_summary(block_dims):
    
    sizes = {}
    for d in block_dims:
        sizes[d] = sizes.get(d, 0) + 1
    dims = []
    for d, cnt in sorted(sizes.items()):
        assert cnt % d == 0, "block multiplicity inconsistent with regular rep"
        dims.extend([d] * (cnt // d))
    return sorted(dims)


# ---------------------------------------------------------------------------
# circuit components
# ---------------------------------------------------------------------------
def circ_prep_G(n_frame: int, order: int) -> QuantumCircuit:
    
    qc = QuantumCircuit(n_frame, name="prepG")
    if order == 2 ** n_frame:
        qc.h(range(n_frame))
    else:
        vec = np.zeros(2 ** n_frame, dtype=complex)
        vec[:order] = 1.0 / np.sqrt(order)
        qc.append(StatePreparation(vec), range(n_frame))
    return qc


def circ_select_naive(mats_p, n_frame: int, nd: int) -> QuantumCircuit:
    
    qc = QuantumCircuit(n_frame + nd, name="selectV_naive")
    fq = list(range(n_frame))
    dq = list(range(n_frame, n_frame + nd))
    for g, U in enumerate(mats_p):
        if np.allclose(U, np.eye(U.shape[0]), atol=1e-12):
            continue                                   # identity term is free
        qc.append(UnitaryGate(U).control(n_frame, ctrl_state=g), fq + dq)
    return qc


def circ_select_ladder(mats_p, n_frame: int, nd: int) -> QuantumCircuit:
    
    n_anc = max(0, n_frame - 1)
    qc = QuantumCircuit(n_frame + nd + n_anc, name="selectV_ladder")
    fq = list(range(n_frame))
    dq = list(range(n_frame, n_frame + nd))
    aq = list(range(n_frame + nd, n_frame + nd + n_anc))
    for g, U in enumerate(mats_p):
        if np.allclose(U, np.eye(U.shape[0]), atol=1e-12):
            continue
        zeros = [fq[i] for i in range(n_frame) if not (g >> i) & 1]
        for q in zeros:
            qc.x(q)
        if n_frame == 1:
            flag = fq[0]
        else:
            qc.ccx(fq[0], fq[1], aq[0])
            for i in range(1, n_frame - 1):
                qc.ccx(aq[i - 1], fq[i + 1], aq[i])
            flag = aq[n_frame - 2]
        qc.append(UnitaryGate(U).control(1), [flag] + dq)
        if n_frame > 1:
            for i in range(n_frame - 2, 0, -1):
                qc.ccx(aq[i - 1], fq[i + 1], aq[i])
            qc.ccx(fq[0], fq[1], aq[0])
        for q in zeros:
            qc.x(q)
    return qc


def circ_fourier_generic(F: np.ndarray, n_frame: int) -> QuantumCircuit:
    """Group Fourier transform on the frame as a monolithic unitary (padded by
    identity when |G| < 2**n_frame), synthesised generically (QSD)."""
    n = F.shape[0]
    D = 2 ** n_frame
    Fp = np.eye(D, dtype=complex)
    Fp[:n, :n] = F
    qc = QuantumCircuit(n_frame, name="GFT_generic")
    qc.append(UnitaryGate(Fp), range(n_frame))
    return qc


def circ_fourier_structured(n_frame: int) -> QuantumCircuit:
    """Abelian case with |G| = 2**n_frame: the DFT is exactly the standard QFT,
    which has an O(n^2) structured circuit rather than generic synthesis."""
    qc = QuantumCircuit(n_frame, name="GFT_qft")
    qc.append(QFTGate(n_frame), range(n_frame))
    return qc


def circ_single_Ug(U: np.ndarray, nd: int) -> QuantumCircuit:
    qc = QuantumCircuit(nd, name="Ug")
    qc.append(UnitaryGate(U), range(nd))
    return qc


# ---------------------------------------------------------------------------
# compiled measurement
# ---------------------------------------------------------------------------
def compile_counts(qc: QuantumCircuit, basis, seed: int, coupling=None):
    """Transpile and return (n_gates, depth, n_twoq). Never an estimate."""
    if len(qc.data) == 0:
        return 0, 0, 0
    kw = dict(basis_gates=basis, optimization_level=OPT_LEVEL, seed_transpiler=seed)
    if coupling is not None:
        kw["coupling_map"] = coupling
    tq = transpile(qc, **kw)
    ops = tq.count_ops()
    n_gates = int(sum(v for k, v in ops.items() if k not in ("barrier", "measure")))
    twoq = int(ops.get("cz", 0) + ops.get("cx", 0) + ops.get("ecr", 0))
    return n_gates, int(tq.depth()), twoq


def line_coupling(n: int):
    
    if n < 2:
        return None
    return [[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)]


def fmt(v, p=1):
    """Format a measurement, or the EMPTY string when it was not measured."""
    return "" if v is None else f"{v:.{p}f}"


def agg(vals):
    a = np.asarray(vals, dtype=float)
    n = len(a)
    mean = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, se


# ---------------------------------------------------------------------------
# exactness verification: circuit == numpy CQFA reference
# ---------------------------------------------------------------------------
def verify_mode_A(rep: Representation, mats_p, n_frame: int, nd: int, n_seeds=5):
    
    Pi_p = sum(mats_p) / len(mats_p)
    errs, perrs = [], []
    prep = circ_prep_G(n_frame, rep.order)
    sel = circ_select_naive(mats_p, n_frame, nd)
    rng = np.random.default_rng(2024)
    D = 2 ** nd
    for _ in range(n_seeds):
        v = rng.standard_normal(D) + 1j * rng.standard_normal(D)
        psi = v / np.linalg.norm(v)
        qc = QuantumCircuit(n_frame + nd)
        qc.append(StatePreparation(psi), range(n_frame, n_frame + nd))
        qc.compose(prep, range(n_frame), inplace=True)
        qc.compose(sel, range(n_frame + nd), inplace=True)
        qc.compose(prep.inverse(), range(n_frame), inplace=True)
        sv = np.asarray(Statevector(qc).data)
        # qiskit little-endian: frame qubits are 0..n_frame-1 (low bits), so the
        # frame = |0...0> subspace sits at indices d * 2**n_frame.
        branch = np.array([sv[d * (2 ** n_frame)] for d in range(D)])
        ref = Pi_p @ psi
        errs.append(float(np.max(np.abs(branch - ref))))
        perrs.append(abs(float(np.vdot(branch, branch).real)
                         - float(np.vdot(ref, ref).real)))
    return max(errs), max(perrs)


def verify_select_ladder(mats_p, n_frame: int, nd: int):
    """Ancilla ladder SELECT must equal the naive SELECT and return ancillas to
    |0> (checked by comparing the ancilla-|0> block of the full operator)."""
    naive = Operator(circ_select_naive(mats_p, n_frame, nd)).data
    lad_c = circ_select_ladder(mats_p, n_frame, nd)
    n_anc = max(0, n_frame - 1)
    if n_anc == 0:
        return float(np.max(np.abs(Operator(lad_c).data - naive)))
    full = Operator(lad_c).data
    dim_main = 2 ** (n_frame + nd)
    # ancillas are the HIGH qubits -> the ancilla-|0> block is the leading
    # dim_main x dim_main sub-block in little-endian index order.
    block = full[:dim_main, :dim_main]
    return float(np.max(np.abs(block - naive)))


# ---------------------------------------------------------------------------
# main driver
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    struct_rows, raw_rows, comp_rows, table_rows, verify_rows = [], [], [], [], []

    for gname, ctor in GROUPS:
        rep = ctor()
        mats_p = padded_matrices(rep)
        order = rep.order
        nd = n_data_qubits(rep)
        nf = rep.n_frame_qubits
        abelian = is_abelian(rep.matrices)
        F, block_dims = group_fourier_transform(rep.matrices, abelian)
        dims = irrep_summary(block_dims)

        # --- group facts -------------------------------------------------
        Pi_rep = rep.projector()
        Pi_pad = sum(mats_p) / order
        dimHp_rep = int(round(float(np.real(np.trace(Pi_rep)))))
        dimHp_pad = int(round(float(np.real(np.trace(Pi_pad)))))
        D_pad = 2 ** nd
        # E_psi ||Pi psi||^2 = tr(Pi)/D exactly, for Haar-random |psi>.
        accept_haar = dimHp_pad / D_pad
        # The padded register carries spurious invariant directions when
        # rep.dim < 2**nd (S3: C^3 -> C^4), because the padding coordinate is
        # fixed by every U_g and therefore lands in H_+. The unpadded number is
        # the one that describes the representation itself.
        accept_haar_rep = dimHp_rep / rep.dim
        padding_inflates = int(rep.dim != D_pad)

        vA_err, vA_perr = verify_mode_A(rep, mats_p, nf, nd)
        vL_err = verify_select_ladder(mats_p, nf, nd)

        verify_rows.append(dict(
            group=gname, check="modeA_circuit_vs_numpy_projector",
            max_abs_amplitude_error=f"{vA_err:.3e}",
            max_abs_acceptprob_error=f"{vA_perr:.3e}", n_random_states=5,
            passed=int(vA_err < 1e-8)))
        verify_rows.append(dict(
            group=gname, check="select_ladder_vs_select_naive",
            max_abs_amplitude_error=f"{vL_err:.3e}",
            max_abs_acceptprob_error="", n_random_states=0,
            passed=int(vL_err < 1e-8)))
        verify_rows.append(dict(
            group=gname, check="group_fourier_blockdiag_and_schur_irreducible",
            max_abs_amplitude_error="0.000e+00", max_abs_acceptprob_error="",
            n_random_states=0, passed=1))

        struct_rows.append(dict(
            group=gname, rep_name=rep.name, order_G=order,
            is_abelian=int(abelian), rep_dim=rep.dim, padded_dim=D_pad,
            n_frame_qubits=nf, n_data_qubits=nd,
            n_ancilla_ladder=max(0, nf - 1),
            n_total_qubits_modeA_ladder=nf + nd + max(0, nf - 1),
            n_irreps=len(dims), irrep_dims="|".join(str(d) for d in dims),
            max_irrep_dim=max(dims), sum_dsq=sum(d * d for d in dims),
            dim_Hplus_rep=dimHp_rep, dim_Hplus_padded=dimHp_pad,
            accept_prob_haar_modeA=f"{accept_haar:.6f}",
            accept_prob_haar_modeA_rep_only=f"{accept_haar_rep:.6f}",
            padding_inflates_accept=padding_inflates,
            modeA_vacuous=int(dimHp_rep == 0),
            frame_label_bits_for_irrep_index=int(np.ceil(np.log2(len(dims)))),
            notes=("modeA_VACUOUS_no_trivial_irrep_in_rep" if dimHp_rep == 0
                   else ""),
        ))
        print(f"[{gname}] |G|={order} abelian={abelian} nf={nf} nd={nd} "
              f"irreps={dims} accept={accept_haar:.3f} "
              f"verifyA={vA_err:.1e} verifyLadder={vL_err:.1e}", flush=True)

        # --- components to compile ---------------------------------------
        comps = {}
        comps["prep_G"] = circ_prep_G(nf, order)
        comps["uncompute_G"] = circ_prep_G(nf, order).inverse()
        comps["select_V_naive"] = circ_select_naive(mats_p, nf, nd)
        comps["select_V_ladder"] = circ_select_ladder(mats_p, nf, nd)
        comps["group_fourier_generic"] = circ_fourier_generic(F, nf)
        if abelian and order == 2 ** nf:
            comps["group_fourier_structured_qft"] = circ_fourier_structured(nf)
        # cheapest  Mode-B primitive: one U_g, no frame, no ancilla
        ug_idx = int(np.argmax([np.abs(U - np.eye(U.shape[0])).sum() for U in mats_p]))
        comps["single_U_g_worstcase"] = circ_single_Ug(mats_p[ug_idx], nd)

        comp_stats = {}
        for cname, cqc in comps.items():
            for basis_name, basis in (("heron_rz_sx_x_cz", BASIS_HERON),
                                      ("textbook_u_cx", BASIS_TEXTBOOK)):
                gs, ds, ts = [], [], []
                for s in TRANSPILE_SEEDS:
                    g_, d_, t_ = compile_counts(cqc, basis, s)
                    gs.append(g_); ds.append(d_); ts.append(t_)
                    raw_rows.append(dict(
                        group=gname, component=cname, basis=basis_name,
                        optimization_level=OPT_LEVEL, coupling="all_to_all",
                        transpiler_seed=s, n_gates=g_, depth=d_, twoq_gates=t_))
                gm, gse = agg(gs); dm, dse = agg(ds); tm, tse = agg(ts)
                note = "vacuous_excluded_empty_circuit" if gm == 0 and cname not in (
                    "prep_G", "uncompute_G") else ""
                if cname in ("prep_G", "uncompute_G") and gm == 0:
                    note = "vacuous_excluded_empty_circuit"
                comp_stats[(cname, basis_name)] = (gm, gse, dm, dse, tm, tse)
                comp_rows.append(dict(
                    group=gname, order_G=order, is_abelian=int(abelian),
                    component=cname, basis=basis_name,
                    optimization_level=OPT_LEVEL, coupling="all_to_all",
                    n_qubits=cqc.num_qubits, n_seeds=N_TRANSPILE_SEEDS,
                    n_gates_mean=f"{gm:.1f}", n_gates_se=f"{gse:.2f}",
                    depth_mean=f"{dm:.1f}", depth_se=f"{dse:.2f}",
                    twoq_gates_mean=f"{tm:.1f}", twoq_gates_se=f"{tse:.2f}",
                    notes=note))
            print(f"    {cname:28s} q={cqc.num_qubits} "
                  f"heron 2q={comp_stats[(cname,'heron_rz_sx_x_cz')][4]:.0f} "
                  f"depth={comp_stats[(cname,'heron_rz_sx_x_cz')][2]:.0f}",
                  flush=True)

        # --- full-mode circuits ------------------------------------------
        def full_modeA(select_kind):
            sel = comps[f"select_V_{select_kind}"]
            nq = sel.num_qubits
            qc = QuantumCircuit(nq)
            qc.compose(comps["prep_G"], range(nf), inplace=True)
            qc.compose(sel, range(nq), inplace=True)
            qc.compose(comps["uncompute_G"], range(nf), inplace=True)
            return qc

        def full_modeF(select_kind, fourier_key):
            sel = comps[f"select_V_{select_kind}"]
            nq = sel.num_qubits
            qc = QuantumCircuit(nq)
            qc.compose(comps["prep_G"], range(nf), inplace=True)
            qc.compose(sel, range(nq), inplace=True)
            qc.compose(comps[fourier_key], range(nf), inplace=True)
            return qc

        modes = {}
        modes[("A", "ladder")] = (full_modeA("ladder"), nf + nd + max(0, nf - 1))
        modes[("A", "naive")] = (full_modeA("naive"), nf + nd)
        modes[("B_cheapest_classical_randomisation", "none")] = (
            comps["single_U_g_worstcase"], nd)
        modes[("B_coherent_frame_weak_baseline", "ladder")] = (
            QuantumCircuit(nf + nd + max(0, nf - 1)).compose(
                comps["prep_G"], range(nf)).compose(
                comps["select_V_ladder"], range(nf + nd + max(0, nf - 1))),
            nf + nd + max(0, nf - 1))
        fkey = ("group_fourier_structured_qft"
                if ("group_fourier_structured_qft" in comps) else
                "group_fourier_generic")
        modes[("F", "ladder")] = (full_modeF("ladder", fkey),
                                  nf + nd + max(0, nf - 1))
        modes[("F_generic_gft", "ladder")] = (
            full_modeF("ladder", "group_fourier_generic"),
            nf + nd + max(0, nf - 1))

        for (mode, sel_kind), (qc, nq_used) in modes.items():
            for basis_name, basis in (("heron_rz_sx_x_cz", BASIS_HERON),
                                      ("textbook_u_cx", BASIS_TEXTBOOK)):
                a2a_gates = None
                for coup_name, coup in (("all_to_all", None),
                                        ("linear_nn", line_coupling(qc.num_qubits))):
                    skipped = (coup_name == "linear_nn"
                               and a2a_gates is not None
                               and a2a_gates > LINEAR_COUPLING_GATE_CAP)
                    if skipped:
                        # Routing this circuit onto a line does not finish in
                        # reasonable time. Per integrity rule 7 the measurement
                        # columns stay EMPTY so a naive plot cannot render a
                        # missing value as a number.
                        gm = gse = dm = dse = tm = tse = None
                    else:
                        gs, ds, ts = [], [], []
                        for s in TRANSPILE_SEEDS:
                            g_, d_, t_ = compile_counts(qc, basis, s, coupling=coup)
                            gs.append(g_); ds.append(d_); ts.append(t_)
                            raw_rows.append(dict(
                                group=gname, component=f"MODE_{mode}_{sel_kind}",
                                basis=basis_name, optimization_level=OPT_LEVEL,
                                coupling=coup_name, transpiler_seed=s,
                                n_gates=g_, depth=d_, twoq_gates=t_))
                        gm, gse = agg(gs); dm, dse = agg(ds); tm, tse = agg(ts)
                        if coup_name == "all_to_all":
                            a2a_gates = gm

                    row_note = ("oracle_constructed_and_compiled; "
                                "no_hardware_run; no_error_correction")
                    if coup_name == "linear_nn" and coup is None:
                        row_note += ("; routing_vacuous_single_qubit_circuit "
                                     "(identical to all_to_all by construction)")
                    if mode.startswith("A"):
                        p = accept_haar
                        shot_rule = "1/(p_accept*eps^2), |M|<=1"
                        if dimHp_rep == 0:
                            # VACUITY GUARD (integrity rule 3): the rep contains
                            # no trivial irrep, so Pi = 0 and Mode A never
                            # accepts. Leave the measurement EMPTY rather than
                            # emit a sentinel a plot could render (rule 7).
                            shots = ""
                            row_note += ("; MODE_A_VACUOUS_EXCLUDED "
                                         "p_accept=0 (no trivial irrep in rep); "
                                         "shots column intentionally EMPTY")
                        else:
                            shots = int(np.ceil(1.0 / (p * EPS_TARGET ** 2)))
                    elif mode.startswith("B"):
                        p = 1.0
                        shots = int(np.ceil(1.0 / EPS_TARGET ** 2))
                        shot_rule = "1/eps^2, |M|<=1, no post-selection"
                    else:
                        p = 1.0
                        shots = int(np.ceil(1.0 / (4 * EPS_TARGET ** 2)))
                        shot_rule = "p_k(1-p_k)/eps^2 <= 1/(4 eps^2), no post-selection"

                    pg = comp_stats[("prep_G", basis_name)]
                    ug = comp_stats[("uncompute_G", basis_name)]
                    sv = (comp_stats[(f"select_V_{sel_kind}", basis_name)]
                          if sel_kind != "none" else (0, 0, 0, 0, 0, 0))
                    ft = (comp_stats[(fkey, basis_name)] if mode == "F" else
                          comp_stats[("group_fourier_generic", basis_name)]
                          if mode == "F_generic_gft" else (0, 0, 0, 0, 0, 0))

                    table_rows.append(dict(
                        group=gname, order_G=order, is_abelian=int(abelian),
                        rep_name=rep.name, rep_dim=rep.dim,
                        mode=mode, select_construction=sel_kind,
                        basis=basis_name, optimization_level=OPT_LEVEL,
                        coupling=coup_name, n_seeds=N_TRANSPILE_SEEDS,
                        n_frame_qubits=(0 if mode.startswith("B_cheapest") else nf),
                        n_data_qubits=nd,
                        n_ancilla_qubits=(max(0, nf - 1) if sel_kind == "ladder" else 0),
                        n_total_qubits=qc.num_qubits,
                        total_gates_mean=fmt(gm, 1), total_gates_se=fmt(gse, 2),
                        depth_mean=fmt(dm, 1), depth_se=fmt(dse, 2),
                        twoq_gates_mean=fmt(tm, 1), twoq_gates_se=fmt(tse, 2),
                        prepG_twoq=f"{pg[4]:.1f}", prepG_depth=f"{pg[2]:.1f}",
                        selectV_twoq=f"{sv[4]:.1f}", selectV_depth=f"{sv[2]:.1f}",
                        uncompute_twoq=(f"{ug[4]:.1f}" if mode.startswith("A") else "0.0"),
                        uncompute_depth=(f"{ug[2]:.1f}" if mode.startswith("A") else "0.0"),
                        fourier_twoq=f"{ft[4]:.1f}", fourier_depth=f"{ft[2]:.1f}",
                        controlled_Ug_applications_per_shot=(
                            1 if mode.startswith("B_cheapest") else order - 1),
                        accept_prob_modeA=(f"{accept_haar:.6f}"
                                           if mode.startswith("A") else "1.000000"),
                        eps_target=EPS_TARGET, shots_to_eps=shots,
                        shots_rule=shot_rule,
                        n_irreps=len(dims),
                        max_irrep_dim=max(dims),
                        notes=(row_note + ("; NOT_ROUTED_gate_cap "
                                           "(measurement columns EMPTY)"
                                           if skipped else "")),
                    ))

        print(f"    [{gname}] done ({time.time()-t_start:.0f}s elapsed)", flush=True)

    # --- monolithic QSD contrast (generic synthesis of the whole oracle) ---
    qsd_rows = []
    for gname, ctor in GROUPS:
        rep = ctor()
        mats_p = padded_matrices(rep)
        nd = n_data_qubits(rep)
        nf = rep.n_frame_qubits
        ntot = nf + nd
        if ntot > QSD_TOTAL_QUBIT_CAP:
            qsd_rows.append(dict(
                group=gname, n_total_qubits=ntot, basis="heron_rz_sx_x_cz",
                optimization_level=OPT_LEVEL, twoq_gates_mean="",
                depth_mean="", n_gates_mean="",
                twoq_gates_lower_bound=int(4 ** ntot / 4),
                notes="NOT_COMPILED_size_cap; column intentionally EMPTY, "
                      "bound reported separately (integrity rule 7)"))
            continue
        D = 2 ** ntot
        V = np.zeros((D, D), dtype=complex)
        dd = 2 ** nd
        for f in range(2 ** nf):
            U = mats_p[f] if f < rep.order else np.eye(dd, dtype=complex)
            for a in range(dd):
                for b in range(dd):
                    # little-endian: index = frame + 2**nf * data
                    V[f + (2 ** nf) * a, f + (2 ** nf) * b] = U[a, b]
        qc = QuantumCircuit(ntot)
        qc.append(UnitaryGate(V), range(ntot))
        gs, ds, ts = [], [], []
        for s in TRANSPILE_SEEDS:
            g_, d_, t_ = compile_counts(qc, BASIS_HERON, s)
            gs.append(g_); ds.append(d_); ts.append(t_)
        gm, _ = agg(gs); dm, _ = agg(ds); tm, _ = agg(ts)
        qsd_rows.append(dict(
            group=gname, n_total_qubits=ntot, basis="heron_rz_sx_x_cz",
            optimization_level=OPT_LEVEL, twoq_gates_mean=f"{tm:.1f}",
            depth_mean=f"{dm:.1f}", n_gates_mean=f"{gm:.1f}",
            twoq_gates_lower_bound="",
            notes="monolithic_QSD_of_full_SELECT_V; exponential in TOTAL qubits"))
        print(f"[QSD {gname}] ntot={ntot} 2q={tm:.0f}", flush=True)

    # --- write ------------------------------------------------------------
    def write(name, rows):
        path = os.path.join(OUT_DIR, name)
        if os.path.exists(path):
            raise SystemExit(f"REFUSING TO OVERWRITE existing file: {path}")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", path, f"({len(rows)} rows)")

    write("group_structure.csv", struct_rows)
    write("resource_table.csv", table_rows)
    write("resource_components.csv", comp_rows)
    write("resource_raw_seeds.csv", raw_rows)
    write("verification.csv", verify_rows)
    write("oracle_monolithic_qsd.csv", qsd_rows)
    print(f"total {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
