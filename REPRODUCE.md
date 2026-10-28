# Reproducing the reported results

Every command is run from the archive root with the environment installed as
described in `README.md`.

Two levels are available. Rebuilding the figures and tables from the recorded
CSVs takes under a minute and requires no simulation. Re-running the
experiments regenerates those CSVs and takes considerably longer.

---

## Level 1 — rebuild every figure and table from recorded results

```bash
python -m figures.make_all_figures      # all eleven figures -> figures/output/
python -m analysis.build_tables         # result tables -> analysis/output/tables.tex
```

Neither trains a model nor runs a simulation. Output can be compared directly
against `figures/pdf/`, which holds the versions used in the submitted
documents.

### Figures individually

| figure | command | reads |
|---|---|---|
| Figure 1 | `python -m figures.fig01_architecture` | schematic, no data |
| Figure 2 | `python -m figures.fig02_two_copy_separation` | `results/two_copy_separation/` |
| Figure 3 | `python -m figures.fig03_discrete_log` | `results/discrete_log/dlp_wide/dlp_wide_agg.csv` |
| Figure 4 | `python -m figures.fig04_frame_readout` | `results/fisher_information/mode_comparison.csv` |
| Figure 5 | `python -m figures.fig05_hardware_mode_f` | `results/ibm_campaign/ibm_modef*/` |
| Figure 6 | `python -m figures.fig06_trainability_law` | `results/trainability_law/trainability_law.csv` |
| Figure 7 | `python -m figures.fig07_representation_dial` | `results/trainability_law/two_knob_sweep.csv` |
| Supp. Figure 1 | `python -m figures.figS1_surplus_dimension` | `results/coherent_surplus/surplus_dimension.csv` |
| Supp. Figure 2 | `python -m figures.figS2_measurement_complexity` | `results/spt_measurement_cost/exponential_separation.csv` |
| Supp. Figure 3 | `python -m figures.figS3_noise_robustness` | `results/robustness/` |
| Supp. Figure 4 | `python -m figures.figS4_generalization` | `results/generalization/learning_curves.csv` |

Diagnostic figures that appear in neither document are retained in
`figures/additional_diagnostics.py`.

---

## Level 2 — re-run the experiments

Each driver writes into `results/<group>/`, overwriting the shipped copy of that
file. Use `git`, a copy of the directory, or `scripts/verify_manifest.py` to
compare against what was shipped.

Every driver takes `--help`.

### The readout basis

The rank-deficiency result: a fixed Fourier readout accesses only
`sum_lambda d_lambda` of the `|G|` frame-state parameters, and a shallow
trainable readout recovers most of the difference on non-abelian groups while
gaining nothing where the deficit is zero.

```bash
python -m experiments.readout_basis.u1_gft_certify              # certify F_G for all twelve groups
python -m experiments.readout_basis.t2_nonabelian_learned_readout
python -m experiments.readout_basis.u1_random_basis_arm         # Haar-random basis control
python -m experiments.readout_basis.u1_shotmatched_nonabelian   # 256-shot control
python -m experiments.readout_basis.u1_abstract_evidence        # collects the reported summary
```

### Two-copy sample complexity

```bash
python -m experiments.two_copy_separation.learning_separation
python -m experiments.two_copy_separation.learning_separation_scaled
python -m experiments.two_copy_separation.fit_growth
python -m experiments.two_copy_separation.scoring_rule_robustness
```

### Discrete log

```bash
python -m experiments.discrete_log.dlp_real
python -m experiments.discrete_log.dlp_baselines_tuned
python -m experiments.discrete_log.dlp_wide            # the 30-prime sweep, slow
python -m experiments.discrete_log.dlp_circuit         # Shor phase estimation, up to 25 qubits
```

### Fisher information and Mode F

```bash
python -m experiments.fisher_information.fourier_mode
python -m experiments.fisher_information.extensions
python -m experiments.fisher_information.shift_recovery_cost
```

### Trainability

```bash
python -m experiments.trainability_law.run_scaling
python -m experiments.trainability_law.trainability_law
python -m experiments.trainability_law.two_knob_sweep
python -m experiments.trainability_law.equivariant_trained
python -m experiments.trainability_law.run_certificate
```

### Coherent surplus

```bash
python -m experiments.coherent_surplus.surplus_dimension
python -m experiments.coherent_surplus.surplus_span_ranks
python -m experiments.coherent_surplus.surplus_corrected_law
```

### Measurement complexity on symmetry-protected order

```bash
python -m experiments.spt_measurement_cost.run_separation
python -m experiments.spt_measurement_cost.exponential_separation
```

### Generalization, noise, benchmark

```bash
python -m experiments.generalization.learning_curves
python -m experiments.noise_robustness.robustness_sweep
python -m experiments.noise_robustness.robustness_noisy_task
python -m experiments.multitask_benchmark.run_benchmarks
```

### Hardware

The hardware drivers submit to IBM Quantum and require credentials. The API
token is never read from or written to this archive: authentication is resolved
by `QiskitRuntimeService()` from the environment variable `QISKIT_IBM_TOKEN` or
a saved Qiskit configuration.

```bash
python -m experiments.hardware_ibm.ibm_campaign.runner --help
python -m experiments.hardware_ibm.ibm_campaign.analyze     # re-analyses recorded counts, no network
python -m experiments.hardware_ibm.equivalence              # simulator/Qiskit cross-check, no network
```

The recorded counts, job identifiers, transpiled depths and qubit layouts from
the reported campaign are in `results/ibm_campaign/`, so the analysis
reproduces without any device access.

---

## Runtime

| stage | cost |
|---|---|
| smoke test | seconds |
| all figures and tables | under a minute |
| test suite | a few minutes |
| most experiment drivers | minutes |
| the 30-prime discrete-log sweep, the fixed-basis pool search, the noise grid | hours |

---

## Determinism

All randomness flows through explicit seed arguments. Every benchmark run
writes a `config.json` next to its outputs recording the representation, model,
task, seed list, train and test sizes, epochs, learning rate, depth and teacher
seed, so any reported value is traceable to the run that produced it.
