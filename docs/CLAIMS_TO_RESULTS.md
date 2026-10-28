# Claims to results

Every number reported in the paper or the supplementary document, the file it
comes from, and the command that regenerates that file.

Paths are relative to the archive root. Commands assume the environment
described in `README.md`. Files listed under `results/curated/` are copies of
the corresponding file in the main results tree, gathered so the evidence
behind cited numbers can be read without navigating the full tree.

---

## The readout basis

The headline result. A fixed Fourier readout of the frame register accesses
only `sum_lambda d_lambda` of the `|G|` real parameters of the frame state; the
remainder `sum_lambda d_lambda (d_lambda - 1)` vanishes exactly when `G` is
abelian.

| claim | file | regenerate with |
|---|---|---|
| Per-group, per-shot-budget accuracy for every readout arm: fixed Fourier, best-of-pool fixed basis, trainable circuit, learned unitary, and the shot-matched control. Nine non-abelian groups and three abelian controls, at shot budgets 16, 64, 256 and exact. This file carries the reported `0.980`, `0.916`, `0.981`, `0.960` and `0.728`. | `results/curated/readout_basis/u1_shotmatched_summary.csv` (per-cell detail in `u1_shotmatched_percell.csv`) | `python -m experiments.readout_basis.u1_shotmatched_nonabelian` |
| Trainable circuit readout against the fixed Fourier family, per group and task | `results/curated/readout_basis/t2_nonabelian_learned_readout.csv`, summarised in `t2_nonabelian_summary.csv` | `python -m experiments.readout_basis.t2_nonabelian_learned_readout` |
| Best-of-pool sweep over fixed Fourier bases | `results/curated/readout_basis/u1_pool_sweep.csv` | `python -m experiments.readout_basis.u1_shotmatched_nonabelian` |
| Untrained Haar-random basis arm, and its collapse at 256 shots | `results/curated/readout_basis/u1_random_basis_arm.csv` | `python -m experiments.readout_basis.u1_random_basis_arm` |
| Gain tracks `1 - sum_lambda d_lambda / |G|` at `r = 0.97`, `p = 1.6e-5`; `r = 0.83` against a single fixed basis; the rival regressor `1/|G|` carries the opposite sign | `results/curated/readout_basis/u1_mechanism.csv`, `u1_mechanism_regressors.csv` | `python -m experiments.readout_basis.u1_abstract_evidence` |
| `F_G` certified a generalized Fourier transform for all twelve groups; maximum off-block magnitude `1.8e-13`; irrep dimensions match `12/12` | `results/curated/readout_basis/u1_gft_certification.csv` | `python -m experiments.readout_basis.u1_gft_certify` |

Column names in these files identify the readout arm directly, for example
`mean_acc_test_modeF_fixed`, `mean_acc_test_modeF_bestbasis` and
`mean_acc_test_learned_circuit`, with matching standard errors.

## Two-copy sample complexity

| claim | file | regenerate with |
|---|---|---|
| Coherent two-copy reader reaches the target accuracy with 2 to 4 copies; single-copy shadow learner grows from `64` copies at `n=2` to `16,384` at `n=9` | `results/curated/two_copy_separation/learning_separation.csv` | `python -m experiments.two_copy_separation.learning_separation_scaled` |
| Fitted exponential base `2.49` in sample, `2.39` held out, exactly `2.00` under an unfitted oracle threshold | `results/two_copy_separation/scoring_rule_robustness_fit.csv` | `python -m experiments.two_copy_separation.scoring_rule_robustness` |
| Subsystem purity and state overlap, evaluated for `n <= 3` | `results/curated/two_copy_separation/entanglement.csv`, `overlap.csv` | `python -m experiments.two_copy_separation.learning_separation` |

## Discrete log

| claim | file | regenerate with |
|---|---|---|
| Quantum feature beats every tuned classical model in all 25 (prime, seed) units, by `0.224`–`0.386` per-prime mean and `0.180`–`0.450` per seed | `results/curated/discrete_log/dlp_baselines_tuned.csv` | `python -m experiments.discrete_log.dlp_baselines_tuned` |
| 30-prime sweep to `p = 4001`: quantum `0.96` to `1.00`, best classical approaching `0.51` | `results/curated/discrete_log/dlp_wide_agg.csv` | `python -m experiments.discrete_log.dlp_wide` |
| Shor discrete-log phase estimation recovers `log_g x` with rate `1.0` for `p = 5, 17, 257`, using up to 25 qubits | `results/curated/discrete_log/dlp_circuit.csv` | `python -m experiments.discrete_log.dlp_circuit` |

## Fisher information and Mode F

The seeding audit described in the supplementary affected this experiment. The
module originally derived its per-readout seed from Python's salted string hash,
so its RNG stream varied between processes. It now seeds from a process-stable
hash, and its outputs were regenerated into
`results/fisher_information/shift_recovery_deterministic/`. That regenerated set
is what the reported numbers come from, and it is what `results/curated/`
carries. The pre-regeneration outputs remain in
`results/fisher_information/` for comparison; their fitted exponent is `2.871`
against the regenerated `2.906`.


| claim | file | regenerate with |
|---|---|---|
| Modes A and B stay near chance at `0.55` with feature variance below `1e-32`; Mode F reaches `0.90` to `1.00` for `|G| = 4` to `32` | `results/curated/fisher_information/mode_comparison.csv` | `python -m experiments.fisher_information.fourier_mode` |
| Non-abelian `S_3`: isotypic readout learns permutation parity at `0.99` | `results/curated/fisher_information/nonabelian_s3.csv` | `python -m experiments.fisher_information.extensions` |
| Discrete recovery cost exponent `2.91`, 95% interval `[2.80, 3.01]` (row `F_fixed_product_coh_bestM`, fitted slope `2.906`) | `results/curated/fisher_information/shift_recovery_cost_fits.csv`, copied from `results/fisher_information/shift_recovery_deterministic/` | `python -m experiments.fisher_information.shift_recovery_cost` |

## Trainability

| claim | file | regenerate with |
|---|---|---|
| Equivariant gradient variance stays at `0.05` with `Var . dim H_+ = 0.2` from `n = 3` to `8`; generic ansatz follows `1/4^n` and reaches `9.5e-7` at `n = 10` | `results/curated/trainability_law/trainability_law.csv` | `python -m experiments.trainability_law.trainability_law` |
| Fitted exponents in `dim H_+` range from `-0.9` to `-2.1` across depth, parameter position, observable and cost locality | `results/curated/trainability_law/variance.csv` | `python -m experiments.trainability_law.run_scaling` |
| All four Representation-Dial quadrants are reachable | `results/curated/trainability_law/two_knob_sweep.csv` | `python -m experiments.trainability_law.two_knob_sweep` |
| Equivariant ansatz trains end to end at `0.90 +/- 0.04` over `n = 3` to `5`; at `n = 6` the same budget reaches `0.58` | `results/trainability_law/equivariant_trained.csv` | `python -m experiments.trainability_law.equivariant_trained` |

## Coherent surplus

| claim | file | regenerate with |
|---|---|---|
| Closed form exact across the ten representations of the sweep; eight satisfy the independence hypothesis | `results/curated/coherent_surplus/surplus_dimension.csv` | `python -m experiments.coherent_surplus.surplus_dimension` |
| Ranks identical at tolerances `1e-6`, `1e-9`, `1e-12` across three independent computations | `results/curated/coherent_surplus/surplus_span_ranks.csv` | `python -m experiments.coherent_surplus.surplus_span_ranks` |
| Across nineteen representations under generic `M`, seven fail the closed form and all seven first fail the independence condition; `S_4` on `C^4` gives closed form `-8` against measured ranks `15` and `16` | `results/curated/coherent_surplus/surplus_corrected_law.csv` | `python -m experiments.coherent_surplus.surplus_corrected_law` |

## Measurement complexity on symmetry-protected order

| claim | file | regenerate with |
|---|---|---|
| Single-copy shadow cost follows `3^w / eps^2` with `V_shadow / 3^w` in `0.87`–`1.04` for `n = 4..10`; reaches `5.2e6` copies against `~3` for the coherent readout; ratio `2e6` at `n = 10` | `results/curated/spt_measurement_cost/exponential_separation.csv` | `python -m experiments.spt_measurement_cost.exponential_separation` |
| A single-copy learner given the Pauli string needs `64`–`128` copies over `n = 2..8` | `results/curated/spt_measurement_cost/cstar.csv` | `python -m experiments.spt_measurement_cost.run_separation` |

## Generalization, noise and benchmark

| claim | file | regenerate with |
|---|---|---|
| Sample-efficiency learning curves | `results/curated/generalization/learning_curves.csv` | `python -m experiments.generalization.learning_curves` |
| Across ten channels and six strengths no non-vacuous cell falls more than `0.069` below its ceiling; the channel removes the task signal in `8` of `60` cells | `results/curated/noise_robustness/noisy_task_verdict.csv` | `python -m experiments.noise_robustness.robustness_noisy_task` |
| Mode-F accuracy spans `0.59`–`0.91`; two-copy separation ratio stays at least `1` in every channel | `results/curated/noise_robustness/modef_allnoise.csv`, `twocopy_allnoise.csv` | `python -m experiments.noise_robustness.robustness_sweep` |
| Matched parameter count `12`: CQFA `0.980 +/- 0.019` against classical frame averaging `0.803 +/- 0.027`; `Z2` benchmark `0.973` against `0.763`; synthetic genomics `0.967` | `results/curated/multitask_benchmark/benchmark_master.csv` | `python -m experiments.multitask_benchmark.run_benchmarks` |

## Hardware

| claim | file | regenerate with |
|---|---|---|
| Mode F recovers `8/8` hidden shifts for `N = 4, 8, 16` in each of two runs, mean peak probabilities `0.87` and `0.90`; Mode B stays near uniform at mean total variation `0.02` | `results/curated/hardware/c8_run1_ibm_fez.csv`, `c8_run2_ibm_fez.csv` | `python -m experiments.hardware_ibm.ibm_campaign.analyze` |
| Job identifiers, transpiled depths, qubit layouts and raw counts | `results/ibm_campaign/` | recorded; `analyze` re-derives every statistic without device access |
| Run-to-run reproducibility: `7` of `8` paired arms differ by more than the within-run 95% shot-noise interval | `results/curated/hardware/hardware_run_reproducibility.csv` | `python -m experiments.hardware_ibm.ibm_campaign.hardware_stats` |

---

## Per-run parameters

The benchmark sweep varies its configuration per cell. Each run writes a
`config.json` next to its outputs recording the representation, model, task,
seed list, train and test sizes, epochs, learning rate, depth and teacher seed.
There are 208 such files under `runs/`, and every row of
`results/aggregated/benchmark_master.csv` carries the path of the run that
produced it.

The classical reverse-complement baseline is not run through that harness; its
parameters are fixed in `src/datasets/rc_symmetric_baseline.py`.
