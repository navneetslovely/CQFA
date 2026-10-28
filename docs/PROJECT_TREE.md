# Project tree

Every file in the archive, with a one-line description. Counts are of files
actually present.

## Top level

```
LICENSE                      review-only licence covering evaluation and reproduction
MANIFEST.sha256              SHA-256 of every other file in the archive
README.md                    what CQFA is, how to install it, and the first command to run
REPRODUCE.md                 the command behind every number, table and figure
pyproject.toml               package metadata; enables `pip install -e .`
requirements.txt             supported version ranges
requirements-lock.txt        the exact versions every reported number was produced on
```

## `src/` — the library (14 files)

Everything in the archive imports from here.

```
src/cqfa/
  __init__.py                package entry point; re-exports the public API
  group.py                   finite groups, unitary representations and the group oracle
  frame.py                   the CQFA layer as statevector operations
  readout.py                 the four model variants, sharing one ansatz and one observable
  simulator.py               deterministic statevector simulator with parameter-shift gradients
  dynamical_lie_algebra.py   dynamical Lie algebra primitives and the gradient-variance law
  certificate.py             the Representation Dial certificate: a computable (T, H) pair
  tasks.py                   learning-task construction
  train.py                   seeded training, bootstrap confidence intervals and CSV logging
  verify.py                  numerical verification of the invariance identities
  paths.py                   filesystem locations used by experiments, analysis and figures

src/datasets/
  __init__.py                data generators for the benchmark sweep
  genomics_reverse_complement.py   reverse-complement (Z2) DNA k-mer classification
  rc_symmetric_baseline.py   a classical reverse-complement-symmetric baseline
  DATA_STATEMENT.md          every data source used, and why
```

## `experiments/` — 119 modules in 13 groups

### `readout_basis/` (17) — the headline result

```
t1_abelian_convergence.py         abelian ceiling / convergence check
t1_summarize.py                   aggregates T1 and applies the pre-registered criterion
t1_verify_probe.py                adversarial verification of T1: under-training, optimiser stress, power
t2_nonabelian_learned_readout.py  non-abelian learned readout against fixed Mode F
t2_adversarial_probe.py           tests whether the abelian null is an under-trained optimiser
t2_baseline_deltas.py             the paired comparison against the best fixed Fourier basis
t3_soft_equivariance.py           soft equivariance under symmetry breaking, empirical test
u1_shotmatched_nonabelian.py      shot-matched learned readout vs fixed Mode F on a group ladder
u1_abstract_evidence.py           the two measurements the abstract sentence depends on
u1_gft_certify.py                 certifies F_G is a generalized Fourier transform (complex-irrep safe)
u1_random_basis_arm.py            the untrained Haar-random basis control arm
u2_symmetry_discovery.py          can a trainable frame find the hidden group from data?
u2_oracle_init_diagnostic.py      post-hoc diagnostic on oracle initialisation
u2_restart_collapse_probe.py      whether validation-split selection would have recovered H
u2_readback.py                    reads back the U2 summary; computes nothing new
u3_soft_equivariance_hard.py      soft equivariance, hard non-saturated redesign
__init__.py                       package marker
```

### `hardware_ibm/` (18) — the IBM Heron campaign

```
__init__.py                  hardware-agnostic qiskit instantiation of the CQFA primitive
cqfa_circuit.py              compiles the abstract model (G, U, V, W) to a real circuit
qiskit_cqfa_reference.py     verified reference: CQFA(Z2) as a gate-based qiskit circuit
equivalence.py               the gating cross-check between circuit and statevector
readout.py                   turns shot counts or a statevector into the coherent readout
backends.py                  labelled backend factory; every executor carries its label
noise_study.py               noise study for the symmetry-protected post-selected readout
ibm_runtime_path.py          budget-guarded real-hardware path; off by default
run_hardware.py              single CLI driver for the hardware experiment
_util.py                     run manifest, CSV writing and shared helpers
ibm_campaign/
  __init__.py                the single-job campaign: all seven device experiments
  specs.py                   PUB builders, one per campaign experiment
  runner.py                  batched execution with a hard QPU-seconds gate
  theory.py                  exact ground truth for every campaign experiment
  analyze.py                 raw per-PUB counts into per-experiment CSV rows; offline
  hardware_stats.py          per-job and per-PUB statistics with uncertainty on every value
  campaign.py                CLI for the seven-experiment campaign
```

### `spt_measurement_cost/` (14) — measurement complexity on symmetry-protected order

```
__init__.py                  SPT measurement-complexity separation
spt_model.py                 state preparation and exact operators
spt_physics_reference.py     independently verified reference physics for the cluster-Ising phase
string_family.py             the symmetry-allowed string family that makes the separation provable
readout_coherent.py          coherent two-copy joint readouts, statevector only
readout_shadow.py            single-copy classical-shadow baseline (Huang-Kueng-Preskill 2020)
sample_complexity.py         the C*(n) driver: matched copy budget per arm
exponential_separation.py    the exponential separation on SPT data
run_separation.py            CLI entry point; writes the traceable CSVs
plots.py                     diagnostic figures generated from those CSVs
tests/                       physics, readout and separation correctness gates (18 tests)
```

### `trainability_law/` (12) — gradient variance and the Representation Dial

```
__init__.py                  trainability scaling and the certified Representation Dial
trainability_law.py          the trainability law: Var ~ 1/dim(H_+)
trainability_full.py         full probabilistic model, with ensemble and cost locality pinned down
trainability_optlink.py      links initial gradient variance to optimisation success
analysis.py                  shared deterministic estimators
run_scaling.py               the gradient-variance-vs-qubit-count experiment
run_dla.py                   dynamical-Lie-algebra dimensions vs qubit count
run_certificate.py           computes the (T, H) certificate points
two_knob_sweep.py            shows the Dial's two knobs are independent
equivariant_trained.py       trains the equivariant ansatz end to end
plot_scaling.py              gradient-variance and DLA-dimension diagnostics
plot_dial.py                 the two-dimensional certified Representation Dial
```

### `discrete_log/` (9)

```
__init__.py                  the Liu-Arunachalam-Temme discrete-log concept class
dlog_concept.py              the LAT concept class at a small prime
learners.py                  two log-blind classical encodings, plus the quantum feature map
dlp_real.py                  the concept class against strong tuned classical baselines
dlp_baselines_tuned.py       full cross-validated hyperparameter search for every classical family
dlp_wide.py                  the wide sweep: 30 primes rather than 3
dlp_wide_large.py            pushes the sweep to large primes
dlp_circuit.py               the circuit-level discrete-log feature map (Shor phase estimation)
run_sweep.py                 multi-seed driver with bootstrap confidence intervals
```

### `two_copy_separation/` (7)

```
__init__.py                    a fair sample-complexity separation over physical quantum data
tasks.py                       a suite of two-copy vs single-copy learning separations
learning_separation.py         the unconditional separation against every single-copy strategy
learning_separation_scaled.py  the same task and learners at larger n
fit_growth.py                  fits the growth exponent of the single-copy arm
scoring_rule_robustness.py     robustness of that exponent to the scoring rule
annotate_censoring.py          annotates the sweep with explicit censoring information
```

### `multitask_benchmark/` (7)

```
__init__.py                  multi-dataset benchmark sweep for the CQFA framework
matrix.py                    the declarative (cell x representation x task x model) grid
metrics.py                   post-hoc classification metrics from per-seed outputs
run_benchmarks.py            the single driver for the sweep
resource_accounting.py       builds and transpiles the circuit; prices the group oracle
resource_analysis.py         abelian vs non-abelian scaling from the compiled counts
resource_scaling_model.py    second analysis pass over those counts
```

### `coherent_surplus/` (5)

```
__init__.py                       package marker
surplus_dimension.py              the coherent-surplus expressivity theorem, quantified
surplus_span_ranks.py             certifies the surplus dimension by operator span ranks
surplus_span_ranks_crosscheck.py  independent verification of the rank deficits
surplus_corrected_law.py          tests a corrected closed form for the surplus dimension
```

### `fisher_information/` (5)

```
__init__.py                  package marker
fourier_mode.py              Mode F (frame Fourier / irrep readout) for a cyclic group
extensions.py                two extensions of Mode F, including non-abelian S_3
noise_resilience.py          resilience of the Mode F readout under realistic noise
shift_recovery_cost.py       separates continuous Fisher information from discrete shift recovery
```

### `noise_robustness/` (5)

```
__init__.py                        package marker
noise_lib.py                       single- and few-qubit noise channel library
robustness_sweep.py                sweeps the two headline results across channels and strengths
robustness_noisy_task.py           robustness against tasks the channel provably preserves
robustness_noisy_task_summary.py   aggregates that sweep into the per-task summary
```

### `equivariance/` (2) and `generalization/` (2)

```
equivariance/verify_identities.py    verifies the exact-invariance identities across every representation
generalization/learning_curves.py    the equivariance-to-generalization sample-efficiency advantage
```

### `exploratory/` (17) — studies reporting no result in either document

```
README.md                                    why these are retained and what they showed
copy_fusion/copy_permutation_advantage.py    CQFA with G = the copy-permutation group S_k
copy_fusion/informed_comparator.py           the informed single-copy comparator
copy_fusion/informed_comparator_pooled.py    pools the per-seed cells of that comparator
fusion/unified_advantage.py                  expressivity surplus and measurement complexity together
novelty_gate/learned_frame_reduction.py      structural reduction check, no training
readout_dial/frame_register_circulant.py     ceiling-theorem check for a learnable readout dial
trifecta/__init__.py                         trainability, training and hardness on one instance
trifecta/trifecta.py                         demonstrates all three on the same representation
trifecta/hardness.py                         the rigorous hardness leg
trifecta/hardness_reduction.py               grounds the acceptance-vs-hardness reduction numerically
trifecta/verify_sat_reduction.py             verifies the exact #P gadget behind the value-hardness route
```

## `figures/` — 19 modules, 11 submitted PDFs

```
style.py                        shared publication style; outlines fonts so no Type 3 or CID font ships
make_all_figures.py             builds every figure in the paper and supplementary document
fig01_architecture.py           Figure 1: the CQFA layer
fig02_two_copy_separation.py    Figure 2: copies required to reach the target accuracy
fig03_discrete_log.py           Figure 3: discrete-log comparison over 30 primes
fig04_frame_readout.py          Figure 4: frame-readout head on the hidden-shift task
fig05_hardware_mode_f.py        Figure 5: cross-sector Mode F on IBM Heron
fig06_trainability_law.py       Figure 6: gradient-variance scaling
fig07_representation_dial.py    Figure 7: the Representation Dial
figS1_surplus_dimension.py      Supplementary Figure 1: coherent-surplus dimension
figS2_measurement_complexity.py Supplementary Figure 2: measurement complexity
figS3_noise_robustness.py       Supplementary Figure 3: ten channels by six strengths
figS4_generalization.py         Supplementary Figure 4: sample efficiency
_architecture.py                the architecture schematic renderer
_paper_figures.py               shared renderers for the main-text figures
_fair_and_noise.py              renderers for f_fair and f_noise
_hardware_mode_f.py             renderer for the hardware Mode F figure
additional_diagnostics.py       diagnostic figures not used in either document
pdf/                            the 11 PDFs as submitted, one per figure
```

Running `make_all_figures` writes regenerated PDFs to `figures/output/`, so the
two directories can be compared directly.

## `analysis/` (2)

```
__init__.py                  package marker
build_tables.py              generates the result tables directly from the recorded CSVs
```

## `scripts/` (16)

```
smoke_test.py                end-to-end check that the archive is installed and working
reproduce_all.sh             figures, tables and tests; --full and --with-hardware modes
build_manifest.py            writes MANIFEST.sha256
verify_manifest.py           checks every file against MANIFEST.sha256
campaigns/README.md          how to run a full experiment group
campaigns/01_readout_basis.sh through 11_hardware_analysis.sh
                             one launcher per experiment group, in dependency order
```

## `tests/` (15 files, 134 tests)

```
test_group.py                    groups, representations and the oracle
test_simulator.py                statevector simulator and parameter-shift gradients
test_readout.py                  the four model variants
test_train.py                    seeded training and bootstrap intervals
test_equivariance.py             the invariance identities
test_certificate.py              the (T, H) certificate and the trainability estimators
test_dynamical_lie_algebra.py    DLA and g-purity primitives behind the trainability law
test_discrete_log.py             correctness gates for the discrete-log concept class
test_spt_measurement_cost.py     correctness gate for the measurement-complexity separation
test_benchmarks.py               fast gates for the multi-dataset benchmark sweep
test_reverse_complement.py       reverse-complement equivariance regression
test_hardware_equivalence.py     the gating test: circuit output equals the statevector model
test_hardware_campaign.py        offline gates for the seven-experiment campaign
test_learning_and_trifecta.py    offline gates for the noise and Mode F additions
__init__.py                      package marker
```

A further 18 tests ship beside the code they cover, in
`experiments/spt_measurement_cost/tests/`. The full suite is 152 tests:

```bash
pytest tests/ experiments/spt_measurement_cost/tests/
```

## `docs/` (4)

```
PROJECT_TREE.md              this file
CLAIMS_TO_RESULTS.md         every reported number, its file, and the command that regenerates it
ENVIRONMENT.md               reference hardware, pinned versions, seed methodology
CHECKLIST_MAP.md             where each reproducibility answer is evidenced
```

## `results/` — 966 files

```
RESULTS_SUMMARY.md           narrative summary of the core results
curated/                     43 CSVs: the evidence behind every cited number, by topic
  coherent_surplus/ (3)   discrete_log/ (4)   fisher_information/ (4)
  generalization/ (1)     hardware/ (6)      multitask_benchmark/ (1)
  noise_robustness/ (3)   readout_basis/ (11) spt_measurement_cost/ (3)
  trainability_law/ (4)   two_copy_separation/ (3)
discrete_log/ (572)          includes 540 per-prime shards of the 30-prime sweep
robustness/ (137)            includes 120 per-cell shards of the noise grid
ibm_campaign/ (48)           raw counts, job identifiers, layouts and depths per campaign run
readout_basis/ (39)          the full readout-basis result set
logs/ (22)                   recorded console output from the campaigns
trainability_law/ (18)       gradient-variance and Dial outputs
multitask_benchmark/ (14)    benchmark sweep outputs
moonshot/ (14)               exploratory outputs retained for completeness
fisher_information/ (11)     Mode F and recovery-cost outputs
two_copy_separation/ (9)     separation outputs
spt_measurement_cost/ (8)    measurement-complexity outputs
coherent_surplus/ (5)   copy_fusion/ (4)   hardware/ (4)   trifecta/ (4)
trainability_law_classical/ (3)   aggregated/ (1)   equivariance/ (1)
fusion/ (1)   generalization/ (1)   novelty_gate/ (1)   readout_dial/ (1)
```

## `runs/` — 1042 files, 208 `config.json`

```
benchmarks/B1 through B5     200 run directories from the benchmark sweep
run_dlog_p47_...             the discrete-log run cited in the figures
run_cqfa_z2_rc_... and 8 more   standalone runs across the four model variants
```

Each run directory holds `config.json` (representation, model, task, seed list,
train and test sizes, epochs, learning rate, depth, teacher seed), plus
`train_log.csv`, `eval_log.csv`, `seed_summary.csv` and `final_metrics.csv`.
