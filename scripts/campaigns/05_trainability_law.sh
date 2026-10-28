#!/usr/bin/env bash
# Trainability and the Representation Dial.
# Writes into results/. Run from the archive root.
set -euo pipefail

echo "Trainability and the Representation Dial"
echo "  -> experiments.trainability_law.run_scaling"
python -m experiments.trainability_law.run_scaling "$@"
echo "  -> experiments.trainability_law.trainability_law"
python -m experiments.trainability_law.trainability_law "$@"
echo "  -> experiments.trainability_law.two_knob_sweep"
python -m experiments.trainability_law.two_knob_sweep "$@"
echo "  -> experiments.trainability_law.equivariant_trained"
python -m experiments.trainability_law.equivariant_trained "$@"
echo "  -> experiments.trainability_law.run_certificate"
python -m experiments.trainability_law.run_certificate "$@"
