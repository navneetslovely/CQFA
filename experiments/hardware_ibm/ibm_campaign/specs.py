
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector

from cqfa.group import X as PX, Z as PZ
from ..cqfa_circuit import (
    build_cqfa_circuit, hardware_representation, site_to_qubit,
    _append_basis_rotation, qiskit_to_numpy_amplitudes, pauli_observable,
)
from . import theory as TH


# ---------------------------------------------------------------------------
# PUB container
# ---------------------------------------------------------------------------
@dataclass
class PubSpec:
    """One batched-job PUB: a measured circuit + its shots + exact ground truth."""

    pub_id: str
    experiment: str                 # "C1".."C7"
    circuit: QuantumCircuit
    shots: int
    analysis: str                   # analyzer dispatch key (see analyze.py)
    theory: dict = field(default_factory=dict)   # exact truth + metadata
    meta: dict = field(default_factory=dict)      # extra CSV labels

    def total_shots(self) -> int:
        return int(self.shots)


# ---------------------------------------------------------------------------
# shared gate helpers
# ---------------------------------------------------------------------------
def _rotate_to_pauli_basis_qubit_eq_site(qc, reg, label: dict):
    """Rotate each support qubit into the Z basis to read its Pauli (qubit=site).

    X -> H ; Y -> Sdg then H ; Z -> nothing. Used by C1 where qubit index == site.
    """
    for s, p in label.items():
        if p == "X":
            qc.h(reg[s])
        elif p == "Y":
            qc.sdg(reg[s])
            qc.h(reg[s])
        elif p == "Z":
            pass
        else:
            raise ValueError(f"bad Pauli {p!r}")


def apply_group_element(qc, data, rep, g_index: int, n_data: int):
    """Append the gates for U_g (rep.matrices[g_index]) on the data register.

    Supported: z2_rc (I, X(x)Z). Extend here for other Mode-B groups. g_index=0 is
    always the identity (nothing). Convention: site0 = X, site1 = Z for z2_rc,
    matching ``cqfa_circuit._apply_oracle``.
    """
    if g_index == 0:
        return
    name = rep.name.lower()
    if name.startswith("z2-rc") or name == "z2_rc":
        qc.x(data[site_to_qubit(0, n_data)])   # X on site0
        qc.z(data[site_to_qubit(1, n_data)])   # Z on site1
    else:
        raise ValueError(f"apply_group_element only implements z2_rc; got {rep.name}")


def _data_frame_circuit(n_data, n_frame):
    """Fresh (data, frame) quantum registers + an empty circuit to compose a prep
    onto (mirrors noise_study._prepared_circuits)."""
    d = QuantumRegister(n_data, "data")
    fr = QuantumRegister(n_frame, "frame")
    return QuantumCircuit(d, fr), d, fr


def _numpy_psi_from_prep(prep_fn, n_data) -> np.ndarray:
    """Exact numpy (site0=MSB) data state produced by a shallow prep on n_data
    qubits — used to score Mode-A/Mode-B theory before the frame is attached."""
    d = QuantumRegister(n_data, "data")
    qc = QuantumCircuit(d)
    prep_fn(qc, d)
    sv = np.asarray(Statevector.from_instruction(qc).data, dtype=complex)
    return qiskit_to_numpy_amplitudes(sv, n_data)


# ===========================================================================
# C1 — SPT string-order direct readout (measurement-complexity headline) + shadow contrast arm
# ===========================================================================
def build_c1(sizes=(4, 5, 6), shots=4096, n_shadow=486, shadow_n=4, seed=0):
    """Direct symmetry-aware readout of the SPT string order at several n, plus a
    single-copy random-Pauli shadow arm at ``shadow_n`` to exhibit the ~3^w cost.

    SPT state = H-on-all + CZ-chain (the cluster/graph state, string order +1).
    Trivial state = H-on-all (paramagnet, string order 0).
    """
    specs = []
    for n in sizes:
        label = TH.spt_string_label(n)                 # {site: Pauli}, weight n
        support = sorted(label.keys())
        # C1 uses the qubit==site convention while theory contracts op_on(label) in
        # site0=MSB order; the two agree iff the string is palindromic. The default
        # bulk SPT string (Z Y X..X Y Z) always is — assert it so a future
        # asymmetric interval can't silently mirror the measured operator.
        assert all(label.get(s) == label.get(n - 1 - s) for s in support), \
            "C1 requires a palindromic SPT string for the qubit==site convention"
        for phase in ("spt", "trivial"):
            q = QuantumRegister(n, "q")
            c = ClassicalRegister(n, "c_str")
            qc = QuantumCircuit(q, c)
            for i in range(n):                          # |+>^n
                qc.h(q[i])
            if phase == "spt":                          # + CZ chain -> cluster state
                for i in range(n - 1):
                    qc.cz(q[i], q[i + 1])
            prep = qc.copy()                            # for exact theory (no meas)
            _rotate_to_pauli_basis_qubit_eq_site(qc, q, label)
            qc.measure(q, c)
            s_exact = TH.c1_string_theory(prep, n)
            specs.append(PubSpec(
                pub_id=f"C1_{phase}_n{n}", experiment="C1", circuit=qc, shots=shots,
                analysis="string_parity",
                theory={"string_order": s_exact, "weight": len(support)},
                meta={"n": n, "phase": phase, "support": support,
                      "three_pow_w": 3 ** len(support)},
            ))
    # shadow contrast arm at shadow_n: single-shot random-Pauli snapshots of the
    # cluster state. bases stored so the analyzer can form the shadow estimator.
    n = shadow_n
    label = TH.spt_string_label(n)
    support = sorted(label.keys())
    rng = np.random.default_rng(seed)
    for k in range(n_shadow):
        bases = rng.integers(0, 3, size=n).astype(int)   # 0:X 1:Y 2:Z per qubit
        q = QuantumRegister(n, "q")
        c = ClassicalRegister(n, "c_str")
        qc = QuantumCircuit(q, c)
        for i in range(n):
            qc.h(q[i])
        for i in range(n - 1):
            qc.cz(q[i], q[i + 1])
        for i in range(n):                               # random single-qubit basis
            b = int(bases[i])
            if b == 0:
                qc.h(q[i])
            elif b == 1:
                qc.sdg(q[i]); qc.h(q[i])
        qc.measure(q, c)
        specs.append(PubSpec(
            pub_id=f"C1_shadow_n{n}_k{k}", experiment="C1", circuit=qc, shots=1,
            analysis="shadow_snapshot",
            theory={"string_label": {int(s): label[s] for s in support}},
            meta={"n": n, "bases": [int(b) for b in bases], "support": support,
                  "weight": len(support), "three_pow_w": 3 ** len(support)},
        ))
    return specs


# ===========================================================================
# C2 — two-copy purity via destructive Bell/SWAP (E9 unification)
# ===========================================================================
def build_c2(thetas=(0.0, np.pi / 8, np.pi / 4), shots=8192):
    """1-qubit mixed states rho(t)=diag(cos^2 t,sin^2 t) (purity 1-0.5 sin^2 2t)
    plus one 2-qubit pure state (purity 1), each read by the two-copy Bell test."""
    specs = []
    # --- 1-qubit mixed, 4 qubits: [s1,e1,s2,e2] = 0,1,2,3 ; Bell on (s1,s2) ---
    for t in thetas:
        q = QuantumRegister(4, "q")
        c_sys = ClassicalRegister(2, "c_sys")           # measured pair (a,b)
        c_env = ClassicalRegister(2, "c_env")           # traced out (ignored)
        qc = QuantumCircuit(q, c_sys, c_env)
        s1, e1, s2, e2 = 0, 1, 2, 3
        for s, e in ((s1, e1), (s2, e2)):               # rho = diag(cos^2,sin^2)
            qc.ry(2.0 * float(t), q[s])
            qc.cx(q[s], q[e])
        qc.cx(q[s1], q[s2]); qc.h(q[s1])                # destructive Bell on systems
        qc.measure(q[s1], c_sys[0]); qc.measure(q[s2], c_sys[1])
        qc.measure(q[e1], c_env[0]); qc.measure(q[e2], c_env[1])
        specs.append(PubSpec(
            pub_id=f"C2_mixed_t{t:.3f}", experiment="C2", circuit=qc, shots=shots,
            analysis="bell_purity",
            theory={"purity": TH.c2_purity_theory_mixed(float(t))},
            meta={"kind": "1q_mixed", "theta": float(t), "n_pairs": 1},
        ))
    # --- 2-qubit pure state, 4 qubits: copy1=(0,1), copy2=(2,3); Bell (0,2),(1,3) ---
    q = QuantumRegister(4, "q")
    c_sys = ClassicalRegister(4, "c_sys")
    qc = QuantumCircuit(q, c_sys)

    def _pure_prep(circ, a, b):
        circ.ry(0.7, a); circ.rz(0.4, a); circ.cx(a, b); circ.ry(0.9, b)
    _pure_prep(qc, q[0], q[1])
    _pure_prep(qc, q[2], q[3])
    qc.cx(q[0], q[2]); qc.h(q[0])
    qc.cx(q[1], q[3]); qc.h(q[1])
    qc.measure(q[0], c_sys[0]); qc.measure(q[2], c_sys[1])   # pair 0: a,b
    qc.measure(q[1], c_sys[2]); qc.measure(q[3], c_sys[3])   # pair 1: a,b
    specs.append(PubSpec(
        pub_id="C2_pure_2q", experiment="C2", circuit=qc, shots=shots,
        analysis="bell_purity", theory={"purity": 1.0},
        meta={"kind": "2q_pure", "theta": float("nan"), "n_pairs": 2},
    ))
    return specs


# ===========================================================================
# shared: build a Mode-A (post-selected CQFA) PUB from a shallow data prep
# ===========================================================================
def _mode_a_pub(rep, prep_fn, observable, shots, pub_id, experiment, meta,
                n_layers=0, theta=None):
    """Compose (shallow prep -> CQFA post-select -> observable read) and attach the
    exact f_A / p theory. Reuses build_cqfa_circuit + statevector_readout."""
    n_data = int(round(np.log2(rep.dim)))
    n_frame = rep.n_frame_qubits
    initc, d, fr = _data_frame_circuit(n_data, n_frame)
    prep_fn(initc, d)
    qc_m, _ = build_cqfa_circuit(rep, theta=theta, mode="postselect", measure=True,
                                 n_layers=n_layers, observable=observable)
    qc_s, _ = build_cqfa_circuit(rep, theta=theta, mode="postselect", measure=False,
                                 n_layers=n_layers)
    measured = initc.compose(qc_m)
    statev = initc.compose(qc_s)
    f_exact, p_exact = TH.cqfa_readout_theory(statev, n_data, n_frame, observable)
    obs_sites = sorted(observable.keys())
    return PubSpec(
        pub_id=pub_id, experiment=experiment, circuit=measured, shots=shots,
        analysis="cqfa_postsel",
        theory={"f_A": f_exact, "p": p_exact},
        meta={**meta, "observable_sites": obs_sites, "n_data": n_data,
              "n_frame": n_frame,
              "observable": "".join(observable[s] + str(s) for s in obs_sites)},
    )


# ===========================================================================
# C3 — symmetry-protected post-selection noise suppression (Fig hw)
# ===========================================================================
def _shallow_prep_z2rc(seed):
    def prep(qc, d):
        rng = np.random.default_rng(seed)
        for q in range(len(d)):
            qc.ry(float(rng.uniform(0, np.pi)), d[q])
            qc.rz(float(rng.uniform(0, 2 * np.pi)), d[q])
        qc.cx(d[0], d[1])
    return prep


def build_c3(shots_list=(1024, 4096, 8192), input_seeds=(11, 23)):
    rep = hardware_representation("z2_rc", 2)
    observable = {0: "X", 1: "Z"}                       # commutes with A=X(x)Z
    specs = []
    for si, seed in enumerate(input_seeds):
        for shots in shots_list:
            specs.append(_mode_a_pub(
                rep, _shallow_prep_z2rc(seed), observable, shots,
                pub_id=f"C3_in{si}_s{shots}", experiment="C3",
                meta={"input_id": f"in{si}", "shot_level": shots}))
    return specs


# ===========================================================================
# C4 — coherent surplus: Mode A (coherent) vs Mode B (classical FA)
# ===========================================================================
def _search_surplus_preps(rep, observable, n_data, n_trials=600, seed=7):
    """Deterministically rank shallow z2_rc preps by |f_A - f_B|. Returns the list
    of (score, prep, f_A, f_B, surplus, p) sorted large -> small (valid p only)."""
    rng = np.random.default_rng(seed)
    cands = []
    for _ in range(n_trials):
        angs = rng.uniform(0, 2 * np.pi, size=(n_data, 2))
        use_cx = bool(rng.integers(0, 2))

        def prep(qc, d, angs=angs, use_cx=use_cx):
            for q in range(len(d)):
                qc.ry(float(angs[q, 0]), d[q]); qc.rz(float(angs[q, 1]), d[q])
            if use_cx:
                qc.cx(d[0], d[1])
        psi = _numpy_psi_from_prep(prep, n_data)
        f_A, f_B, surplus, p = TH.surplus_theory(psi, rep, observable, n_data)
        if surplus is None or p < 0.15:                # need a workable accept rate
            continue
        cands.append((abs(surplus), prep, f_A, f_B, surplus, p))
    if not cands:
        raise RuntimeError("no valid surplus prep found")
    cands.sort(key=lambda t: t[0], reverse=True)
    return cands


def build_c4(shots=8192):
    rep = hardware_representation("z2_rc", 2)
    observable = {0: "Z", 1: "X"}          # commutes with A=X(x)Z; surplus ~ <Y0 Y1>
    n_data = 2
    specs = []
    cands = _search_surplus_preps(rep, observable, n_data)
    # (a) max-surplus input; (b) min-surplus input (near-classical control)
    inputs = [("maxsurplus", *cands[0][1:]), ("control", *cands[-1][1:])]

    for input_id, prep, f_A, f_B, surplus, p in inputs:
        # Mode A (coherent, post-selected)
        specs.append(_mode_a_pub(
            rep, prep, observable, shots,
            pub_id=f"C4_{input_id}_modeA", experiment="C4",
            meta={"input_id": input_id, "mode": "A", "surplus_group": input_id,
                  "f_A_theory": f_A, "f_B_theory": f_B, "surplus_theory": surplus}))
        # Mode B arms (classical frame averaging: one direct measurement per U_g)
        for g in range(rep.order):
            d = QuantumRegister(n_data, "data")
            c = ClassicalRegister(n_data, "c_data")
            qc = QuantumCircuit(d, c)
            prep(qc, d)
            apply_group_element(qc, d, rep, g, n_data)
            _append_basis_rotation(qc, d, observable, n_data)
            qc.measure(d, c)
            arm_theory = TH.modeB_arm_theory(
                _numpy_psi_from_prep(prep, n_data), rep.matrices[g], observable, n_data)
            specs.append(PubSpec(
                pub_id=f"C4_{input_id}_modeB_g{g}", experiment="C4", circuit=qc,
                shots=shots, analysis="direct_expval",
                theory={"arm_value": arm_theory},
                meta={"input_id": input_id, "mode": "B", "g": g,
                      "surplus_group": input_id,
                      "observable_sites": sorted(observable.keys()), "n_data": n_data}))
    return specs


# ===========================================================================
# C5 — exact G-equivariance under device noise (Thm 1)
# ===========================================================================
def build_c5(shots=8192, input_seeds=(5, 17)):
    rep = hardware_representation("z2_rc", 2)
    # Must NOT equal A=X0 Z1: that observable pins f_A=1 for every input (A|Pi psi>
    # =|Pi psi>), so the equivariance gap is structurally 0 and cannot discriminate.
    # Z0 X1 commutes with A but is not A, giving input-dependent yet exactly
    # g-invariant f_A -> a gap that genuinely moves if noise breaks equivariance.
    observable = {0: "Z", 1: "X"}
    n_data = 2
    specs = []
    for si, seed in enumerate(input_seeds):
        base_prep = _shallow_prep_z2rc(seed)
        # arm g=0 : psi ; arm g=1 : U_g psi (append A after the prep)
        for g in range(rep.order):
            def prep(qc, d, base_prep=base_prep, g=g):
                base_prep(qc, d)
                apply_group_element(qc, d, rep, g, n_data)
            specs.append(_mode_a_pub(
                rep, prep, observable, shots,
                pub_id=f"C5_in{si}_g{g}", experiment="C5",
                meta={"input_id": f"in{si}", "g": g, "equiv_group": f"in{si}"}))
    return specs


# ===========================================================================
# C6 — Z2xZ2 multi-frame-qubit oracle (|G|=4, 2 frame qubits)
# ===========================================================================
def build_c6(shots=8192):
    rep = hardware_representation("z2xz2", 4)           # 4 data + 2 frame = 6 qubits
    observable = {0: "X"}                               # commutes with the X-strings
    specs = []

    def cluster_prep(qc, d):
        for q in range(len(d)):
            qc.h(d[q])
        for q in range(len(d) - 1):
            qc.cz(d[q], d[q + 1])

    def random_prep(qc, d, seed=31):
        rng = np.random.default_rng(seed)
        for q in range(len(d)):
            qc.ry(float(rng.uniform(0, np.pi)), d[q])
            qc.rz(float(rng.uniform(0, 2 * np.pi)), d[q])
        for q in range(len(d) - 1):
            qc.cx(d[q], d[q + 1])

    for name, prep in (("cluster", cluster_prep), ("random", random_prep)):
        specs.append(_mode_a_pub(
            rep, prep, observable, shots,
            pub_id=f"C6_{name}", experiment="C6",
            meta={"input_id": name}))
    return specs


# ===========================================================================
# C7 — depth-scaling sweep (post-selected error & accept rate vs 2q depth)
# ===========================================================================
def build_c7(shots=4096, layers=(0, 1, 2, 3), seed=41):
    rep = hardware_representation("z2_rc", 2)
    observable = {0: "X", 1: "Z"}
    n_data = 2
    specs = []
    base_prep = _shallow_prep_z2rc(seed)
    rng = np.random.default_rng(seed + 1)
    for L in layers:
        theta = (rng.standard_normal(max(1, L) * n_data * 2) * 0.3) if L > 0 else None
        specs.append(_mode_a_pub(
            rep, base_prep, observable, shots,
            pub_id=f"C7_L{L}", experiment="C7", n_layers=L, theta=theta,
            meta={"input_id": "fixed", "n_layers": L}))
    return specs


# ===========================================================================
# C8 — Mode F (frame Fourier readout) hidden-shift recovery vs blind Mode B
# ===========================================================================
def _c8_phase_oracle(qc, frame, s: int, N: int):
 
    f = len(frame)
    for j in range(f):
        qc.p(2.0 * np.pi * s * (2 ** j) / N, frame[j])


def _c8_iqft(qc, frame):
   
    f = len(frame)
    for i in range(f // 2):
        qc.swap(frame[i], frame[f - 1 - i])
    for j in range(f):
        for m in range(j):
            qc.cp(-np.pi / (2 ** (j - m)), frame[m], frame[j])
        qc.h(frame[j])


def build_c8(frame_shifts=((2, (1, 2, 3)), (3, (1, 3, 6)), (4, (5, 11))), shots=4096):

    specs = []
    for f, shifts in frame_shifts:
        N = 2 ** f
        for s in shifts:
            for mode in ("F", "B"):
                frame = QuantumRegister(f, "frame")
                c = ClassicalRegister(f, "c_frame")
                qc = QuantumCircuit(frame, c)
                for j in range(f):                      # uniform frame superposition
                    qc.h(frame[j])
                _c8_phase_oracle(qc, frame, s, N)       # imprint the hidden shift
                if mode == "F":
                    _c8_iqft(qc, frame)                 # coherent Mode-F readout
                prep = qc.copy()                        # (no measure) for exact theory
                qc.measure(frame, c)
                probs = TH.c8_frame_distribution_theory(prep, f)
                ideal_peak_k = int(np.argmax(probs))
                specs.append(PubSpec(
                    pub_id=f"C8_{mode}_N{N}_s{s}", experiment="C8", circuit=qc,
                    shots=shots, analysis="fourier_peak",
                    theory={"hidden_shift": s, "N": N, "mode": mode,
                            "ideal_peak_k": ideal_peak_k,
                            "ideal_peak_prob": float(np.max(probs)),
                            "ideal_uniform": bool(mode == "B")},
                    meta={"n_frame": f, "mode": mode, "hidden_shift": s, "N": N}))
    return specs


# ===========================================================================
# assemble the full campaign
# ===========================================================================
def build_campaign(which=("C1", "C2", "C3", "C4", "C5", "C6", "C7"), overrides=None):
    """Return the full list of PubSpecs for the requested experiments."""
    overrides = overrides or {}
    builders = {
        "C1": build_c1, "C2": build_c2, "C3": build_c3, "C4": build_c4,
        "C5": build_c5, "C6": build_c6, "C7": build_c7, "C8": build_c8,
    }
    specs = []
    for key in which:
        specs.extend(builders[key](**overrides.get(key, {})))
    return specs
