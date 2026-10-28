#!/usr/bin/env bash
# The readout-basis result.
# Writes into results/. Run from the archive root.
set -euo pipefail

echo "The readout-basis result"
echo "  -> experiments.readout_basis.u1_gft_certify"
python -m experiments.readout_basis.u1_gft_certify "$@"
echo "  -> experiments.readout_basis.t2_nonabelian_learned_readout"
python -m experiments.readout_basis.t2_nonabelian_learned_readout "$@"
echo "  -> experiments.readout_basis.u1_random_basis_arm"
python -m experiments.readout_basis.u1_random_basis_arm "$@"
echo "  -> experiments.readout_basis.u1_shotmatched_nonabelian"
python -m experiments.readout_basis.u1_shotmatched_nonabelian "$@"
echo "  -> experiments.readout_basis.u1_abstract_evidence"
python -m experiments.readout_basis.u1_abstract_evidence "$@"
