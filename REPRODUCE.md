# Reproducing the reported results

Every command is run from the archive root with the environment installed as
described in `README.md`.

Two levels are available. Rebuilding the figures and tables from the recorded
CSVs takes under a minute and requires no simulation. Re-running the
experiments regenerates those CSVs and takes considerably longer.

---

## Level 1 — rebuild every figure and table from recorded results

```bash
python -m figures.make_all_figures      # the ten data figures -> figures/output/
python -m analysis.build_tables         # result tables -> analysis/output/tables.tex
```

Neither trains a model nor runs a simulation. Output can be compared directly
against `figures/pdf/`, which holds the versions used in the article and its
supplementary material.

### Figures individually

The script names follow an earlier numbering; the first column gives the
article's.

| article | command | file in `figures/pdf/` | reads |
|---|---|---|---|
| Fig. 1 | schematic, not regenerated | `f0_architecture.pdf` | no data |
| Fig. 2 | `python -m figures.figS1_surplus_dimension` | `f9_surplus_dimension.pdf` | `results/coherent_surplus/surplus_dimension.csv` |
| Fig. 3 | `python -m figures.fig02_two_copy_separation` | `f_fair.pdf` | `results/two_copy_separation/` |
| Fig. 4 | `python -m figures.fig03_discrete_log` | `f_dlp_wide.pdf` | `results/discrete_log/dlp_wide/dlp_wide_agg.csv` |
| Fig. 5 | `python -m figures.fig04_frame_readout` | `f_modef.pdf` | `results/fisher_information/mode_comparison.csv` |
| Fig. 6 | `python -m figures.fig05_hardware_mode_f` | `f_modef_hw.pdf` | `results/ibm_campaign/ibm_modef*/` |
| Fig. 7 | `python -m figures.fig06_trainability_law` | `f8_trainability_law.pdf` | `results/trainability_law/trainability_law.csv` |
| Fig. 8 | `python -m figures.fig07_representation_dial` | `f11_two_knob.pdf` | `results/trainability_law/two_knob_sweep.csv` |
| Supplementary Fig. S1 | `python -m figures.figS2_measurement_complexity` | `f2b_exponential_separation.pdf` | `results/spt_measurement_cost/exponential_separation.csv` |
| Supplementary Fig. S2 | `python -m figures.figS4_generalization` | `f10_generalization.pdf` | `results/generalization/learning_curves.csv` |
| Supplementary Fig. S3 | `python -m figures.figS3_noise_robustness` | `f_noise.pdf` | `results/robustness/` |

Table 1 of the article is read from
`results/curated/readout_basis/u4_strong_fourier_sampling.csv`.

Diagnostic figures that appear in neither document are retained in
`figures/additional_diagnostics.py`.

---

## Level 2 — re-run the experiments

Each driver writes into `results/<group>/`, overwriting the shipped copy of that
file; `git diff` shows any difference from what was shipped.

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
python -m experiments.readout_basis.u4_strong_fourier_sampling   # strong vs weak Fourier sampling arms
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
python -m analysis.dlp_summary                         # gap ranges and label agreement quoted in the article
```

### Fisher information and Mode F

```bash
python -m experiments.fisher_information.fourier_mode
python -m experiments.fisher_information.extensions
python -m experiments.fisher_information.shift_recovery_cost
for m in lb_zero_fi_nogo lb_cert la_single_setting_estimator la_cert lc_qcrb_certificate \
         lc_cert ld_access_iff_rank ld_cert le_dlp_reduction le_cert; do
    python -m experiments.fisher_information.certificates.$m   # writes results/fisher_information/certificates/
done
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
python -m experiments.noise_robustness.robustness_noisy_task --tag _v2   # the corrected run the article cites
python -m experiments.noise_robustness.robustness_noisy_task_summary     # channel spread, from the _v2 files
python -m experiments.noise_robustness.robustness_noisy_task_shotsweep   # Mode-F shot-budget sweep
python -m experiments.multitask_benchmark.run_benchmarks
python -m experiments.multitask_benchmark.resource_accounting            # compiled circuit costs
```

### Hardness checks quoted in Supplementary Section S1

```bash
python -m experiments.hardness.hardness_reduction      # writes results/hardness/
python -m experiments.hardness.verify_sat_reduction
```

### Hardware

The hardware drivers submit to IBM Quantum and require credentials. No API key
is stored in this repository; the runner reads it from the environment at run
time, or falls back to a saved Qiskit account:

```bash
export QISKIT_IBM_TOKEN="<your IBM Quantum API key>"
export QISKIT_IBM_INSTANCE="<your IBM Quantum instance CRN>"   # optional
```

```bash
python -m experiments.hardware_ibm.ibm_campaign.runner --help
python -m experiments.hardware_ibm.ibm_campaign.analyze     # re-analyses recorded counts, no network
python -m experiments.hardware_ibm.equivalence              # simulator/Qiskit cross-check, no network
python -m experiments.hardware_ibm.ibm_campaign.hardware_stats  # run-to-run variation and provenance audit, no network
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
| most experiment drivers | minutes |
| the 30-prime discrete-log sweep, the fixed-basis pool search, the noise grid | hours |

---

## Determinism

All randomness flows through explicit seed arguments. Every benchmark run
writes a `config.json` next to its outputs recording the representation, model,
task, seed list, train and test sizes, epochs, learning rate, depth and teacher
seed, so every run is traceable to its parameters.
