#!/usr/bin/env bash
# Fisher information and Mode F.
# Writes into results/. Run from the archive root.
set -euo pipefail

echo "Fisher information and Mode F"
echo "  -> experiments.fisher_information.fourier_mode"
python -m experiments.fisher_information.fourier_mode "$@"
echo "  -> experiments.fisher_information.extensions"
python -m experiments.fisher_information.extensions "$@"
echo "  -> experiments.fisher_information.shift_recovery_cost"
python -m experiments.fisher_information.shift_recovery_cost "$@"
