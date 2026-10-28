#!/usr/bin/env bash
# Two-copy sample complexity.
# Writes into results/. Run from the archive root.
set -euo pipefail

echo "Two-copy sample complexity"
echo "  -> experiments.two_copy_separation.learning_separation"
python -m experiments.two_copy_separation.learning_separation "$@"
echo "  -> experiments.two_copy_separation.learning_separation_scaled"
python -m experiments.two_copy_separation.learning_separation_scaled "$@"
echo "  -> experiments.two_copy_separation.fit_growth"
python -m experiments.two_copy_separation.fit_growth "$@"
echo "  -> experiments.two_copy_separation.scoring_rule_robustness"
python -m experiments.two_copy_separation.scoring_rule_robustness "$@"
