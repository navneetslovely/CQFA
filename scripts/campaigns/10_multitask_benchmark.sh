#!/usr/bin/env bash
# Multi-task benchmark.
# Writes into results/. Run from the archive root.
set -euo pipefail

echo "Multi-task benchmark"
echo "  -> experiments.multitask_benchmark.run_benchmarks"
python -m experiments.multitask_benchmark.run_benchmarks "$@"
