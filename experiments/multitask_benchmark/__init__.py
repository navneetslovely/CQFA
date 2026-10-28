"""
experiments.multitask_benchmark
======================
Multi-dataset benchmark sweep for the CQFA framework.

The single driver `run_benchmarks.py` runs the full (group x task x model) matrix
via cqfa.train.run_experiment, aggregates over >=5 seeds with bootstrap 95% CIs,
and writes ONE tidy master CSV (results/aggregated/benchmark_master.csv) plus
per-cell run folders under runs/benchmarks/.

See README.md for the matrix, the CSV schema, and the per-cell scope ledger.
"""
