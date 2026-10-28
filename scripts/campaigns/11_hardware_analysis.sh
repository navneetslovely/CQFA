#!/usr/bin/env bash
# Re-analyse the recorded hardware campaign (no device access).
# Writes into results/. Run from the archive root.
set -euo pipefail

echo "Re-analyse the recorded hardware campaign (no device access)"
echo "  -> experiments.hardware_ibm.ibm_campaign.analyze"
python -m experiments.hardware_ibm.ibm_campaign.analyze "$@"
echo "  -> experiments.hardware_ibm.ibm_campaign.hardware_stats"
python -m experiments.hardware_ibm.ibm_campaign.hardware_stats "$@"
