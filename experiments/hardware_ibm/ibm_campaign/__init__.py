"""
experiments.hardware_ibm.ibm_campaign
==================================
The SINGLE-JOB IBM hardware campaign: all seven device experiments (C1-C7)
batched into one SamplerV2 job under a hard QPU-seconds gate.

    C1  SPT string-order direct readout (measurement-complexity headline), n = 4,5,6
        + classical-shadow contrast arm at n = 4
    C2  Two-copy purity via destructive Bell/SWAP test (E9 unification)
        + single-copy tomography contrast arm
    C3  Symmetry-protected post-selection noise suppression (Fig hw)
    C4  Coherent surplus: Mode A (coherent) vs Mode B (classical FA)
    C5  Exact G-equivariance under device noise (Thm 1)
    C6  Z2xZ2 multi-frame-qubit oracle (|G| = 4, 2 frame qubits)
    C7  Depth-scaling sweep (post-selected error & accept rate vs 2q depth)

Module map
----------
    theory.py    exact statevector/numpy ground truth for every circuit
    specs.py     PubSpec builders (circuit + shots + theory meta per PUB)
    runner.py    per-PUB-shots execution: Aer ideal / Fake device / gated IBM
    analyze.py   raw counts -> per-experiment CSVs (pure offline re-analysis)
    campaign.py  CLI:  plan | run | analyze

Reproducibility contract
------------------------
* every run writes results/ibm_campaign/<tag>/: raw_counts.json,
  manifest.json (backend, job id, usage, calibration snapshot, versions),
  circuits.qpy + circuits_transpiled.qpy, analysis/*.csv.
* the API token is NEVER written to disk; auth comes from the saved qiskit
  config or the QISKIT_IBM_TOKEN env var only, and the manifest is scrubbed.
* the real-device path requires three locks (CQFA_IBM_RUN=1, --ibm,
  --ibm-confirm) AND the estimated QPU-seconds to fit --budget-seconds.
"""
