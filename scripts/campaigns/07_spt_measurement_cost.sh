#!/usr/bin/env bash
# Measurement complexity on symmetry-protected order.
# Writes into results/. Run from the archive root.
set -euo pipefail

echo "Measurement complexity on symmetry-protected order"
echo "  -> experiments.spt_measurement_cost.run_separation"
python -m experiments.spt_measurement_cost.run_separation "$@"
echo "  -> experiments.spt_measurement_cost.exponential_separation"
python -m experiments.spt_measurement_cost.exponential_separation "$@"
