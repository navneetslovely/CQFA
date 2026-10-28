# Data statement — CQFA

All data used in the paper is either generated deterministically by released code or
constructed from a documented generator. No proprietary or restricted data is used.

## Quantum-data tasks (SPT)
Ground states of the cluster-Ising chain `H(λ) = -Σ Z_{i-1}X_iZ_{i+1} + λ Σ X_iX_{i+1}`
are produced by exact diagonalization (`experiments/e2_spt/`) at labeled points in the
SPT (λ<1) and trivial (λ>1) phases. Fully reproducible from the Hamiltonian; no external
data.

## Group/task benchmark (benchmark)
Learning tasks are generated from fixed group representations and seeded teacher
observables (`cqfa/tasks.py`); each run records its seeds in `config.json`.

## Genomics reverse-complement task
`datasets/genomics_rc.py` builds a reverse-complement-symmetric (ℤ₂) DNA k-mer
classification with a **documented, deterministic generator**: sequences over
{A,C,G,T}; the label is invariant under reverse-complement by construction (a motif is
counted on either strand), matching the biological reverse-complement symmetry. Seeds
are fixed; the generator and its parameters are in the module. This is a controlled task
for measuring equivariance-as-inductive-bias; it uses no restricted biological data.

## Discrete-log task
The Liu–Arunachalam–Temme concept class is generated over `Z_p^*` at small prime `p`
(`experiments/e3_dlog/dlog_concept.py`); fully reproducible.

## Licensing
All generated data and code are released under the repository's license for
independent reproduction.
