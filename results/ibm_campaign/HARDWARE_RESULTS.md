# CQFA on IBM hardware — results (2026-07-13; Mode F added 2026-07-18)

All seven core experiments (C1–C7) plus the **Mode F** headline (C8, added
2026-07-18, and independently **reproduced** as a second device run) executed on IBM
Quantum Heron r2 devices. Four batched `SamplerV2` jobs; **95 quantum-seconds total**
of billed QPU time. Reproduce analysis offline from the saved raw counts:
`python -m experiments.hardware.ibm_campaign.campaign analyze --tag <tag>`.

| Job | tag | backend | job_id | PUBs | actual QPU-s |
|-----|-----|---------|--------|------|-------------|
| probe | `ibm_c3_probe` | ibm_marrakesh | `d99qv7af47jc73a9eri0` | 6 | 9 |
| main  | `ibm_main` | ibm_fez | `d99r060tcv6s73dnmbr0` | 512 | 48 |
| **Mode F** | `ibm_modef` | ibm_fez | `d9d6ltkinv1c73aokueg` | 16 | **19** |
| **Mode F (repro)** | `ibm_modef2` | ibm_fez | `d9dd6paneu4c739n3brg` | 16 | **19** |

Estimate was 11.3 + 62.4 + 24.9 + 24.9 = 124 s; actual 9 + 48 + 19 + 19 = **95 s**
(estimator conservative). Auth was in-memory only (token never written to disk);
token/CRN scan across all artifacts: CLEAN.

## Results (measured vs exact)

**C1 — SPT string order (exponential-separation headline).** Symmetry-aware direct
readout of the weight-n Y-decorated string:
- SPT (cluster state): n=4 → +0.848, n=5 → −0.791, n=6 → +0.752 (exact ±1)
- trivial (paramagnet): ≈ +0.02 (exact 0) at all n
- clean SPT-vs-trivial separation on hardware, degrading gracefully with n (depth).
- single-copy shadow arm: 5/486 hits, rate 0.0103 vs 1/3^w = 1/81 = 0.0123 —
  the ~3^n single-copy cost, on hardware (illustration; the quantitative variance
  law is the simulation).

**C2 — two-copy purity via destructive Bell/SWAP.** Tr(ρ²): 0.989 / 0.741 / 0.506
vs exact 1.00 / 0.75 / 0.50 (1-qubit mixed sweep); 2-qubit pure 0.937 vs 1.0. The
copy-permutation purity primitive is essentially exact on hardware.

**C3 — symmetry-protected post-selection (the headline figure), on ibm_marrakesh.**
Post-selection vs raw readout error, exact f_A = 1.0:
- input in0: post-sel err 0.03 vs raw err ≈0.18 (≈6× suppression), accept ≈0.91
- input in1: post-sel err 0.03–0.10 vs **raw err ≈1.13** (≈14× suppression), accept ≈0.43
- replaces the previously-simulated FakeBackend figure with a genuine device result.

**C4 — coherent surplus (Mode A − Mode B).** max-surplus input: −0.903 vs exact
−0.992; near-symmetric control: +0.039 vs 0. The coherent surplus — interference
classical frame-averaging cannot represent — **survives on hardware** as a large,
sign-definite signal cleanly separated from the control.

**C5 — exact G-equivariance under device noise (Thm 1).** With a non-degenerate
observable (Z₀X₁), f_A is input-dependent (+0.43, −0.14) yet g-invariant: gap
|f_A(ψ) − f_A(U_gψ)| = 0.007 and 0.010 — equivariance holds within shot noise, and
the metric now genuinely discriminates.

**C6 — Z₂×Z₂ two-frame-qubit oracle.** ⌈log₂|G|⌉=2 frame register, |G|=4: cluster
0.013 (exact 0), random −0.597 vs exact −0.655 (accept ≈0.28). The general
finite-group frame register works on hardware, not just the trivial Z₂ case.

**C7 — post-selected error vs 2q-depth.** Ansatz layers L=0..3; post-selection keeps
the estimate legible where the raw readout collapses (L=0: post-sel err 0.13 vs raw
1.19; L=3: 0.18 vs 0.97), accept ≈0.41 across depth.

**C8 — Mode F (frame Fourier readout) on hardware: the headline advantage, on a
real device (`ibm_fez`, job `d9d6ltkinv1c73aokueg`, 19 QPU-s).** Frame Fourier
sampling recovers a hidden shift `s` (the Mode-F coherent readout), versus the
decohered Mode-B control (classical frame averaging, same prep with NO inverse-QFT)
which is provably blind. Frame register = ⌈log₂ N⌉ qubits; the phase oracle is
single-qubit, so the only 2q gates are the inverse-QFT (2q-depth 5 at N=4 → 22 at
N=16), and the graceful degradation tracks that depth:

| N (frame) | Mode F: shifts recovered | Mode F mean peak-prob | Mode B TV-from-uniform | Mode B peak-prob |
|-----------|--------------------------|-----------------------|------------------------|------------------|
| 4  | **3/3** | 0.902 | 0.015 | ≈0.259 (≈1/4) |
| 8  | **3/3** | 0.872 | 0.023 | ≈0.135 (≈1/8) |
| 16 | **2/2** | 0.807 | 0.020 | ≈0.068 (≈1/16) |
| **all** | **8/8** | **0.867** | **0.019 (blind)** | **0.165 (≈1/N)** |

- Every Mode-F arm recovered the exact hidden shift on hardware (`peak_k = s`,
  peak-prob 0.78–0.91 across N=4,8,16); the coherent Fourier peak survives real
  Heron noise cleanly.
- The Mode-B control is uniform to TV ≈ 0.02 — classical frame averaging carries
  **zero** information about the shift the Fourier readout resolves. This is the
  Mode-F-vs-Mode-B blindness separation, on a real device.
- Ideal correctness gate: 8/8 exact recovery, peak-prob 1.000 (offline). FakeFez
  device-noise dry run predicted 8/8 at mean peak-prob 0.925; the real device gave
  8/8 at 0.867 — consistent with the noise model.
- **Reproduced** on a second independent device run (`ibm_modef2`, job
  `d9dd6paneu4c739n3brg`, 19 QPU-s) on a **different physical-qubit layout**
  ([0,1,2,3,123,136,143,144] vs run-1's [0,1,2,3,73,74,75,136,141,142,143]):
  Mode F again **8/8** recovered at mean peak-prob **0.897** (N=4 0.885, N=8 0.943,
  N=16 0.847), Mode B again blind (TV 0.018). Per-shift peak-probs agree with run-1
  within device noise → the Mode-F/Mode-B separation is robust across layouts, not a
  transpilation artifact.
- Source: `results/ibm_campaign/ibm_modef/analysis/c8.csv` (+ `ibm_modef2/` repro);
  circuits in `specs.build_c8`; exact theory `theory.c8_frame_distribution_theory`.

## Device provenance (per manifest.json)
- ibm_fez used physical qubits [0,1,2,3,4,5,25,26,37,123,124,125,136,142,143];
  12 two-qubit cz edges recorded with real errors (e.g. 0–1: 0.0096, 1–2: 0.0024).
- Each run dir holds raw_counts.json, pub_index.json, manifest.json (job_id, usage,
  full calibration), circuits.qpy + circuits_transpiled.qpy, analysis/*.csv.
