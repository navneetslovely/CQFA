#!/usr/bin/env bash
# Discrete-log concept class.
# Writes into results/. Run from the archive root.
set -euo pipefail

echo "Discrete-log concept class"
echo "  -> experiments.discrete_log.dlp_real"
python -m experiments.discrete_log.dlp_real "$@"
echo "  -> experiments.discrete_log.dlp_baselines_tuned"
python -m experiments.discrete_log.dlp_baselines_tuned "$@"
echo "  -> experiments.discrete_log.dlp_wide"
python -m experiments.discrete_log.dlp_wide "$@"
echo "  -> experiments.discrete_log.dlp_circuit"
python -m experiments.discrete_log.dlp_circuit "$@"
