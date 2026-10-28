"""
Trainability scaling + the certified Representation Dial.

Modules
-------
analysis          : shared estimators (wide_params, gradient-variance,
                    bootstrap CI, exp/poly fits).
run_scaling       : gradient-variance-vs-qubit-count sweep -> variance.csv, fits.csv
run_dla           : dynamical-Lie-algebra dimensions vs n -> dla.csv
run_certificate   : the (T, H) certificate points + decoupling grid -> certificate CSVs
plot_scaling      : figures for the variance sweep + DLA + 1/dim(g) cross-check
plot_dial         : the 2D Representation-Dial figure (T vs H, four quadrants)

See README.md for the PROVEN-vs-EVIDENCE ledger.
"""
