#!/usr/bin/env bash
# Noise robustness.
# Writes into results/. Run from the archive root.
set -euo pipefail

echo "Noise robustness"
echo "  -> experiments.noise_robustness.robustness_sweep"
python -m experiments.noise_robustness.robustness_sweep "$@"
echo "  -> experiments.noise_robustness.robustness_noisy_task"
python -m experiments.noise_robustness.robustness_noisy_task "$@"
