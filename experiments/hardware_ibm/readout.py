"""
experiments.hardware_ibm.readout
=============================
Turn shot counts (or an exact statevector) into the CQFA coherent readout.

Two realisations, matching the spec:

  (a) POST-SELECTION (headline, == Mode A). Keep only shots with frame = 0..0.
      On that sub-ensemble estimate <Z_{site0}>; the per-shot value is
      (-1)^{measured bit of site0}. Acceptance rate estimates p = ||Pi psi||^2.

  (b) RAW (control, no post-selection). Same <Z_{site0}> estimate but over ALL
      shots ignoring the frame — the baseline the post-selection is compared to.

  (c) HADAMARD-TEST (diagnostic). <Z_anc> = Re<psi|A|psi>, an interference
      numerator, NOT f_A. Cross-checked against numpy in equivalence.py.

Bitstring convention
--------------------
``backends._join_register_counts`` yields counts keyed
"<reg0_bits> <reg1_bits> ..." in ClassicalRegister-declaration order, with a
sentinel ``__reg_order__`` naming the registers. Each register's bitstring is
MSB-first (qiskit default): the leftmost char is the highest-index qubit of that
register. In the CQFA circuit the data register declares qubit q = numpy site
(n_data-1-q); so within ``c_data`` the leftmost char is data qubit (n_data-1) =
numpy site 0. We therefore read site s from character index (n_data-1-s) counted
from the RIGHT, i.e. bitstring[::-1][ site_to_qubit(s) ].
"""
from __future__ import annotations

import numpy as np

from .cqfa_circuit import site_to_qubit, qiskit_to_numpy_amplitudes


# ---------------------------------------------------------------------------
# counts -> per-register bitstrings
# ---------------------------------------------------------------------------
def _split_counts(counts: dict):
    """Yield (reg_dict, shots) where reg_dict maps reg_name -> {bits: n}.

    Handles the space-joined counts produced by
    ``backends._join_register_counts``. reg_dict preserves per-register marginal
    but we need joint (frame,data) correlation, so we instead return the raw
    joint counts plus the register order.
    """
    reg_order = counts.get("__reg_order__")
    if reg_order is None:
        raise ValueError("counts missing __reg_order__ sentinel")
    joint = {k: v for k, v in counts.items() if k != "__reg_order__"}
    total = sum(joint.values())
    return joint, reg_order, total


def _reg_index(reg_order, name):
    return reg_order.index(name)


def _bit_of_site(data_bits: str, site: int, n_data: int) -> int:
    """Extract the measured bit of numpy `site` from a data-register bitstring.

    ``data_bits`` is MSB-first over data qubits (leftmost = data qubit n_data-1).
    Reversing gives qubit-index order; data qubit for site s is
    ``site_to_qubit(s, n_data)``.
    """
    rev = data_bits[::-1]                      # rev[q] = bit of data qubit q
    q = site_to_qubit(site, n_data)
    return int(rev[q])


# ---------------------------------------------------------------------------
# post-selected + raw readout from counts
# ---------------------------------------------------------------------------
def readout_from_counts(counts: dict, n_data: int, n_frame: int,
                        observable_site: int = 0, observable_sites=None):
    """Estimate f_A (post-selected) and f_raw from measured counts.

    Returns a dict with keys:
      f_hat_postsel, f_hat_raw, accept_rate, n_accepted, n_total.

    The observable is a product of single-qubit +/-1 eigenvalues over
    ``observable_sites`` (a list of numpy site indices); each site must have had
    its measurement-basis rotation applied by ``build_cqfa_circuit`` for
    non-Z Paulis. If ``observable_sites`` is None, uses the single site
    ``observable_site`` (Z-type). Post-selected ensemble = shots with all frame
    bits 0; raw ensemble = all shots (frame ignored).
    """
    sites = observable_sites if observable_sites is not None else [observable_site]
    joint, reg_order, total = _split_counts(counts)
    ci_data = _reg_index(reg_order, "c_data")
    ci_frame = _reg_index(reg_order, "c_frame")

    acc_num = 0.0            # sum of +/-1 over accepted shots
    acc_n = 0               # accepted shot count
    raw_num = 0.0
    raw_n = 0
    for key, n in joint.items():
        parts = key.split(" ")
        data_bits = parts[ci_data]
        frame_bits = parts[ci_frame]
        eig = 1.0
        for s in sites:
            eig *= 1.0 - 2.0 * _bit_of_site(data_bits, s, n_data)   # (-1)^bit
        raw_num += eig * n
        raw_n += n
        if all(c == "0" for c in frame_bits):
            acc_num += eig * n
            acc_n += n
    f_postsel = (acc_num / acc_n) if acc_n > 0 else float("nan")
    f_raw = (raw_num / raw_n) if raw_n > 0 else float("nan")
    return {
        "f_hat_postsel": f_postsel,
        "f_hat_raw": f_raw,
        "accept_rate": acc_n / total if total else float("nan"),
        "n_accepted": acc_n,
        "n_total": total,
    }


# ---------------------------------------------------------------------------
# exact statevector readout (ground truth for the noise study & equivalence)
# ---------------------------------------------------------------------------
def statevector_readout(statevector, n_data: int, n_frame: int, M: np.ndarray,
                        observable_in_numpy_order: bool = True, tol: float = 1e-12):
    """Exact post-selected coherent readout from a statevector (no shots).

    ``statevector`` is a qiskit Statevector or 1D array over (data (low),
    frame (high)) qubits. Post-select frame=0..0 -> the first 2**n_data
    amplitudes are the data block; renormalise, reorder to numpy convention, and
    read <M>. Returns (f_A, p). If ``M`` is None, uses Z on site 0.
    """
    amp = np.asarray(getattr(statevector, "data", statevector), dtype=complex)
    dimD = 2 ** n_data
    data_block = amp[:dimD]                    # frame = 0..0 (frame are high bits)
    p = float(np.real(data_block.conj() @ data_block))
    if p < tol:
        return None, p
    chi_q = data_block / np.sqrt(p)
    chi_np = qiskit_to_numpy_amplitudes(chi_q, n_data)
    if M is None:
        from cqfa.simulator import local_z
        M = local_z(n_data, 0)
    f = float(np.real(chi_np.conj() @ M @ chi_np))
    return f, p


# ---------------------------------------------------------------------------
# Hadamard-test readout (diagnostic)
# ---------------------------------------------------------------------------
def hadamard_from_counts(counts: dict) -> float:
    """<Z_anc> from a single-ancilla Hadamard test = Re<psi|A|psi>.

    Expects the ancilla classical register 'c_anc' (1 bit). Returns the
    interference numerator (NOT f_A).
    """
    joint, reg_order, total = _split_counts(counts)
    ci = _reg_index(reg_order, "c_anc")
    num = 0.0
    for key, n in joint.items():
        bit = int(key.split(" ")[ci])
        num += (1.0 - 2.0 * bit) * n
    return num / total if total else float("nan")
