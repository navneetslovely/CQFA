"""
experiments.spt_measurement_cost
==================
SPT measurement-complexity separation.

On non-re-preparable SPT quantum data, a coherent 2-copy joint readout (the CQFA
frame register realized as a controlled-SWAP / Bell measurement, group G_copy = S2)
estimates the discriminating nonlinear functional with O(1)-O(n) copies, while any
single-copy + classical-postprocessing strategy needs 2^Omega(n) copies. The
separation is a property of the TASK (nonlinear functional / unknown string), not of
any single known operator -- the known-string control stays flat to prove this.

Modules:
  spt_model          -- H(lam) with the XX competitor (corrected physics), ground
                        states, exact cluster state, K_i stabilizers, string order.
  readout_coherent   -- 2-copy statevector readouts (NO 4^n objects): <P>^2, purity,
                        Bell sampler, explicit frame-register swap test.
  readout_shadow     -- single-copy classical shadows (HKP20), direct-Pauli control.
  string_family      -- S_string enumeration (Option A/B) and the nonlinear functional.
  sample_complexity  -- C*(n) driver: budget sweep, crossing, seeds, bootstrap CIs.
  run_separation             -- CLI entry point; writes the traceable CSVs.
  plots              -- money/ratio/physics figures from the CSVs.

Provable vs empirical: see README.md §"What is provable vs empirical".
"""
from __future__ import annotations

from . import spt_model, readout_coherent, readout_shadow, string_family, sample_complexity

__all__ = [
    "spt_model",
    "readout_coherent",
    "readout_shadow",
    "string_family",
    "sample_complexity",
]
