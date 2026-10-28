
from __future__ import annotations

import numpy as np

from .cqfa_circuit import site_to_qubit, qiskit_to_numpy_amplitudes


# ---------------------------------------------------------------------------
# counts -> per-register bitstrings
# ---------------------------------------------------------------------------
def _split_counts(counts: dict):

    reg_order = counts.get("__reg_order__")
    if reg_order is None:
        raise ValueError("counts missing __reg_order__ sentinel")
    joint = {k: v for k, v in counts.items() if k != "__reg_order__"}
    total = sum(joint.values())
    return joint, reg_order, total


def _reg_index(reg_order, name):
    return reg_order.index(name)


def _bit_of_site(data_bits: str, site: int, n_data: int) -> int:

    rev = data_bits[::-1]                      # rev[q] = bit of data qubit q
    q = site_to_qubit(site, n_data)
    return int(rev[q])


# ---------------------------------------------------------------------------
# post-selected + raw readout from counts
# ---------------------------------------------------------------------------
def readout_from_counts(counts: dict, n_data: int, n_frame: int,
                        observable_site: int = 0, observable_sites=None):

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
