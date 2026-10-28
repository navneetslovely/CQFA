"""
experiments.hardware_ibm.backends
==============================
LABELLED backend factory. Every executor carries a human-readable ``label`` that
is written into every CSV row, so no result ever appears without its backend
provenance (integrity rule). Four kinds:

    "aer_statevector"  -> Aer:statevector(exact)   exact, no shots (equivalence gate)
    "aer_noiseless"    -> Aer:noiseless            AerSimulator, shots
    "aer_depol:<p>"    -> Aer:depol(p=<p>)         depolarizing NoiseModel
    "fake:<name>"      -> Fake:<name>              FakeBackendV2 device-noise model

FakeBackends are *device-noise models*, NOT the live devices — captions must say
"device-noise model (Fake...)", never "run on manila". These run by default (no
credentials, no budget). The real IBM path lives in ``ibm_runtime_path.py`` and
is gated off by default.
"""
from __future__ import annotations

from dataclasses import dataclass

from qiskit import transpile
from qiskit.quantum_info import Statevector


# ---------------------------------------------------------------------------
# runner wrapper
# ---------------------------------------------------------------------------
@dataclass
class BackendRunner:
    """Uniform interface over the different execution paths.

    Attributes
    ----------
    label : str
        Human-readable backend label written into CSVs.
    noise_model : str
        Short tag for the CSV ``noise_model`` column ("none", "depolarizing",
        "device").
    noise_strength : float
        Numeric strength for the CSV (depol p; 0.0 for noiseless; -1 for device).
    kind : str
        One of {"statevector", "sampler"}.
    """

    label: str
    noise_model: str
    noise_strength: float
    kind: str
    _backend: object = None          # AerSimulator (sampler paths) or None
    _sampler: object = None          # qiskit_aer SamplerV2 (sampler paths)
    _seed: int | None = None

    # -- exact statevector (equivalence gate only) --------------------------
    def statevector(self, circuit) -> Statevector:
        if self.kind != "statevector":
            raise RuntimeError(f"backend {self.label} is not a statevector backend")
        return Statevector.from_instruction(circuit)

    # -- shot-based sampling ------------------------------------------------
    def transpile(self, circuits, seed=None):
        """Transpile to the backend basis/coupling; returns transpiled circuits."""
        if self._backend is None:
            return circuits
        return transpile(circuits, self._backend, optimization_level=1,
                         seed_transpiler=self._seed if seed is None else seed)

    def run_sampler(self, circuits, shots: int, seed=None):
        """Run a list of measured circuits; return a list of counts dicts.

        Counts keys are qiskit's default classical-register-joined bitstrings
        (register order matches how the circuit declared its ClassicalRegisters;
        readout.py parses them by register, not by guessing positions).
        """
        if self.kind != "sampler":
            raise RuntimeError(f"backend {self.label} is not a sampler backend")
        single = not isinstance(circuits, (list, tuple))
        circs = [circuits] if single else list(circuits)
        tqc = self.transpile(circs, seed=seed)
        job = self._sampler.run(tqc, shots=shots)
        result = job.result()
        counts_list = []
        for pub in result:
            # SamplerV2: pub.data has one BitArray per ClassicalRegister.
            data = pub.data
            reg_names = list(data.keys()) if hasattr(data, "keys") else None
            if reg_names is None:
                # fall back: single default register
                counts_list.append(pub.data.meas.get_counts())
            else:
                counts_list.append(_join_register_counts(data, reg_names))
        return counts_list[0] if single else counts_list

    def transpiled_2q_depth(self, circuit, seed=None) -> int:
        """Two-qubit depth after transpilation (explains noise degradation)."""
        tqc = self.transpile(circuit, seed=seed)
        c = tqc[0] if isinstance(tqc, list) else tqc
        try:
            return c.depth(lambda instr: instr.operation.num_qubits == 2)
        except Exception:
            return int(c.depth())


def _join_register_counts(data, reg_names) -> dict:
    """Merge per-register BitArrays into one counts dict keyed 'reg1 reg0' order.

    We key the combined string as '<c_frame> <c_data>' by joining registers in a
    fixed, documented order so readout.py can split unambiguously. To keep it
    simple and register-name driven, we return a dict keyed by a tuple-string
    "<name>=<bits>;..." — but for compatibility with count-summing we produce the
    space-joined qiskit-style string in registration order.
    """
    # Build per-shot combined outcomes to preserve joint correlations.
    per_reg = {name: data[name] for name in reg_names}
    n_shots = None
    for name in reg_names:
        n_shots = per_reg[name].num_shots
        break
    # Reconstruct joint counts: iterate each register's get_counts is not enough
    # (loses correlation), so use array_of_ints per shot.
    import numpy as np
    arrays = {}
    for name in reg_names:
        ba = per_reg[name]
        # bit_array -> 2D array (shots, bits) MSB-first per qiskit convention
        arrays[name] = ba
    combined = {}
    # Use the raw per-shot bit arrays.
    shot_ints = {name: per_reg[name].array for name in reg_names}
    n_shots = next(iter(per_reg.values())).num_shots
    # Each BitArray.array is (shots, ceil(bits/8)) bytes; get_bitstrings is easier.
    bitstrings = {name: per_reg[name].get_bitstrings() for name in reg_names}
    for i in range(n_shots):
        key = " ".join(bitstrings[name][i] for name in reg_names)
        combined[key] = combined.get(key, 0) + 1
    combined["__reg_order__"] = reg_names  # sentinel so readout knows the order
    return combined


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def get_backend(kind: str, seed: int | None = None) -> BackendRunner:
    """Return a LABELLED :class:`BackendRunner` for the requested kind.

    kind:
      "aer_statevector"            exact statevector (no shots)
      "aer_noiseless"              AerSimulator, shots
      "aer_depol:<p>"              depolarizing noise, 1q err p/10, 2q err p
      "fake:manila" / "fake:sherbrooke"   FakeBackendV2 device-noise model
    """
    key = kind.lower()

    if key in ("aer_statevector", "statevector", "exact"):
        return BackendRunner(label="Aer:statevector(exact)", noise_model="none",
                             noise_strength=0.0, kind="statevector", _seed=seed)

    if key in ("aer_noiseless", "noiseless"):
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2
        backend = AerSimulator()
        sampler = SamplerV2(seed=seed)
        return BackendRunner(label="Aer:noiseless", noise_model="none",
                             noise_strength=0.0, kind="sampler",
                             _backend=backend, _sampler=sampler, _seed=seed)

    if key.startswith("aer_depol"):
        p = float(key.split(":", 1)[1]) if ":" in key else 1e-2
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(p / 10, 1),
                                       ["ry", "rz", "h", "x", "sx"])
        nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), ["cx", "cz"])
        backend = AerSimulator(noise_model=nm)
        sampler = SamplerV2(seed=seed, options={"backend_options": {"noise_model": nm}})
        return BackendRunner(label=f"Aer:depol(p={p:g})", noise_model="depolarizing",
                             noise_strength=p, kind="sampler",
                             _backend=backend, _sampler=sampler, _seed=seed)

    if key.startswith("fake"):
        name = key.split(":", 1)[1] if ":" in key else "manila"
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2
        fake = _fake_backend(name)
        backend = AerSimulator.from_backend(fake)
        # Extract the noise model so the SamplerV2 also applies it.
        from qiskit_aer.noise import NoiseModel
        nm = NoiseModel.from_backend(fake)
        sampler = SamplerV2(seed=seed, options={"backend_options": {"noise_model": nm}})
        return BackendRunner(label=f"Fake:{name}", noise_model="device",
                             noise_strength=-1.0, kind="sampler",
                             _backend=backend, _sampler=sampler, _seed=seed)

    raise ValueError(f"unknown backend kind '{kind}'")


def _fake_backend(name: str):
    """Load a FakeBackendV2 by short name."""
    from qiskit_ibm_runtime import fake_provider as fp
    key = name.lower()
    if key == "manila":
        return fp.FakeManilaV2()
    if key == "sherbrooke":
        return fp.FakeSherbrooke()
    if key == "lima":
        return fp.FakeLimaV2()
    # Generic: try Fake<Name>V2 (older naming) then Fake<Name> (Heron r2, no suffix)
    cap = name.capitalize()
    cls = getattr(fp, f"Fake{cap}V2", None) or getattr(fp, f"Fake{cap}", None)
    if cls is None:
        raise ValueError(f"unknown fake backend '{name}'")
    return cls()
