# CQFA — core results summary

This file covers the core invariance, surplus and separation results. Results added
later — the two-copy and discrete-log comparisons, Mode F, circuit-level Shor discrete
log to 25 qubits, the ten-channel noise sweep and the IBM Heron campaign — are recorded
in the corresponding subdirectories of `results/`. `docs/CLAIMS_TO_RESULTS.md` maps
every number reported in the paper and supplementary to its file and the command that
regenerates it.

Every number below is produced by a module in `experiments/` and traces to a file in
`results/`. Seeds are fixed, so re-running reproduces the values exactly.

Environment: see `docs/ENVIRONMENT.md`.

---

## Theorem 1 — exact equivariance for arbitrary G
*Source:* `cqfa_core_verify.py` → `results/cqfa_core_verify_output.txt`

Maximum invariance error `|f(g·x) − f(x)|` over inputs and group elements:

| Group | base model (non-equivariant) | classical FA (Mode A) | **CQFA (Mode B, coherent)** |
|---|---|---|---|
| ℤ₂ (reverse-complement), \|G\|=2 | 2.295 | 0.0 | **0.0** |
| C₄ (cyclic), \|G\|=4 | 2.878 | 2.22×10⁻¹⁶ | **0.0** |
| S₃ (non-abelian), \|G\|=6 | 4.220 | 4.44×10⁻¹⁶ | **8.88×10⁻¹⁶** |

→ CQFA and classical FA are invariant to machine precision; the unconstrained base
model is not. Confirms Theorem 1 across abelian and non-abelian groups.

## Theorem 2 — coherent–classical separation
*Source:* `cqfa_core_verify.py` → `results/cqfa_core_verify_output.txt`

| Group | (a) symmetric input ⇒ surplus≈0 | (b) M∝I ⇒ surplus≈0 | (c) max\|surplus\| (generic M) |
|---|---|---|---|
| ℤ₂ | 1.11×10⁻¹⁵ | 2.22×10⁻¹⁶ | **2.866** |
| C₄ | 1.33×10⁻¹⁵ | 0.0 | **4.000** |
| S₃ | 7.77×10⁻¹⁶ | 1.11×10⁻¹⁶ | **0.686** |

→ (a) frame entanglement is a *necessary* resource (zero asymmetry ⇒ zero surplus);
(b) a non-trivial observable is also required (non-sufficiency); (c) the
coherent surplus is large for generic observables. The order parameter is the frame
asymmetry q = 1−‖Πψ‖² (monotone), **not** the entanglement entropy (non-monotone) —
see the (d) co-emergence table in the raw output.

## Theorem 2(d) — a clear quantum advantage over classical frame averaging
*Source:* `cqfa_advantage.py` → `results/cqfa_advantage_output.txt`

Learning task whose label is a cross-orbit coherence (ℤ₂ on ℂ⁸, balanced classes):

| Model | Test accuracy |
|---|---|
| **Quantum / CQFA (coherent, 1 frame qubit)** | **0.995** |
| Classical frame averaging (avg diagonal output) | 0.810 |
| Classical FA, stronger (all diagonal orbit outputs) | 0.808 |
| Chance baseline | 0.507 |

Plus **26 information-theoretic collision pairs** (inputs with ~identical classical-FA
features but opposite labels) → a hard ceiling below 100% for any classical-FA
classifier, while the coherent feature separates them. The advantage is provable, not
a tuning artifact.

---

### scope (do not over-read)
These are **statevector verifications of the theory** and a **constructed** advantage
task. They establish: exact any-group equivariance, the coherent-vs-classical
separation, and that the coherent model accesses signal classical frame averaging
cannot. They do **not** include the genomics, SPT/quantum-data or discrete-log
learning experiments, the performance-model fits, or the hardware campaign; those are
recorded under their own subdirectories of `results/`.
At the scales here everything is classically simulable; **no runtime quantum-advantage
is claimed.**
