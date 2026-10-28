"""
experiments.hardware_ibm
=====================
Hardware-agnostic qiskit instantiation of the CQFA primitive (hardware experiment).

The SAME abstract (G, U, V, W) CQFA model that ``cqfa.frame`` /
``cqfa.readout`` realise in pure numpy compiles here to an explicit gate-based
qiskit circuit. The gating claim (see ``equivalence.py``) is that the qiskit *statevector*
coherent readout reproduces the numpy ``cqfa_readout`` / ``CQFAModel.feature``
to < 1e-9 for G in {Z2, Z2xZ2}. No quantum advantage is claimed — this experiment
is a portability + noise-robustness certificate.

Public entry points
-------------------
- :func:`cqfa_circuit.build_cqfa_circuit`  — abstract rep -> QuantumCircuit
- :func:`cqfa_circuit.hardware_representation` — named reps usable on hardware
- :mod:`backends`     — LABELLED backend factory (Aer / depol / Fake / IBM)
- :mod:`readout`      — sampler-based post-selected + Hadamard-test readout
- :mod:`equivalence`  — statevector cross-check (GATING)
- :mod:`noise_study`  — multi-backend, multi-seed noise sweep -> CSV
- :mod:`run_hardware` — CLI driver
- :mod:`ibm_runtime_path` — budget-guarded real-hardware path (off by default)
"""
