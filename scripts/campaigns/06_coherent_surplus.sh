#!/usr/bin/env bash
# Coherent-surplus dimension.
# Writes into results/. Run from the archive root.
set -euo pipefail

echo "Coherent-surplus dimension"
echo "  -> experiments.coherent_surplus.surplus_dimension"
python -m experiments.coherent_surplus.surplus_dimension "$@"
echo "  -> experiments.coherent_surplus.surplus_span_ranks"
python -m experiments.coherent_surplus.surplus_span_ranks "$@"
echo "  -> experiments.coherent_surplus.surplus_corrected_law"
python -m experiments.coherent_surplus.surplus_corrected_law "$@"
