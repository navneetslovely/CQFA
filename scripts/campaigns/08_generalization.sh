#!/usr/bin/env bash
# Sample efficiency.
# Writes into results/. Run from the archive root.
set -euo pipefail

echo "Sample efficiency"
echo "  -> experiments.generalization.learning_curves"
python -m experiments.generalization.learning_curves "$@"
