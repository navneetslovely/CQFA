# CQFA — Coherent Quantum Frame Averaging

Source code, recorded results and figure generators for the submission
*Coherent Quantum Frame Averaging: A Symmetrization Primitive with Separable Trainability and Hardness Controls*.

Everything needed to reproduce, conduct and analyse the reported experiments is
here: the library, every experiment driver, the analysis code, the plotting
code that produced each figure, all recorded results, and the per-run
configuration of every benchmark cell.

---

## What CQFA is

A finite group `G` acts on a data register through a unitary representation
`U_g`. A frame register of `ceil(log2 |G|)` ancillas holds the uniform
superposition over group elements, and a controlled oracle
`V = sum_g |g><g| (x) U_g` applies the representation conditioned on the frame.
The group average `Pi = (1/|G|) sum_g U_g` projects onto the invariant subspace
`H_+`.

Measuring the frame selects the readout:

| mode | frame measurement | reads |
|---|---|---|
| **A** | post-select on the uniform state | `f_A = <Pi psi\| M \|Pi psi> / \|\|Pi psi\|\|^2` |
| **B** | decohere the register | `f_B = (1/\|G\|) sum_g <psi\| U_g^dag M U_g \|psi>` |
| **F** | measure in the irrep basis | an irrep label on every shot |

Both scalar readouts are exactly `G`-invariant. Their difference is the
coherent surplus: the off-diagonal interference that output averaging cannot
represent.

---

## Install

Python 3.11 to 3.14; 3.14.5 is the reference interpreter.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt     # exact versions used for the reported numbers
pip install -e .
```

`requirements.txt` gives supported version ranges instead of exact pins. The
library itself is pure NumPy; the scientific stack is needed by the experiment
drivers, the analysis code and the figure scripts. The quantum stack is needed
only for circuit construction and the hardware experiments — every simulated
result reproduces without it.

## First command

```bash
python scripts/smoke_test.py
```

Under a minute. It builds a group, checks the projector and oracle, evaluates
both readouts, confirms the invariance identities hold to machine precision,
and confirms the recorded results are present.

## Rebuild every figure

```bash
python -m figures.make_all_figures
```

Reads the recorded CSVs and writes all eleven figures into `figures/output/`.
It trains nothing and simulates nothing. The versions used in the submitted
documents are in `figures/pdf/`, so the two directories can be compared
directly.

## Run the tests

```bash
pytest tests/ experiments/spt_measurement_cost/tests/
```

152 tests: 134 across `tests/`, plus 18 module tests that ship beside the
symmetry-protected-order code. The hardware tests make no network calls.

---

## Layout

| directory | contents |
|---|---|
| `src/cqfa/` | the library — everything imports from here |
| `src/datasets/` | deterministic data generators |
| `experiments/` | drivers, one package per result, writing into `results/` |
| `analysis/` | table generation from recorded CSVs |
| `figures/` | one script per figure, plus the submitted PDFs in `figures/pdf/` |
| `results/` | 211 result CSVs and 709 JSON records; `results/curated/` holds the 41 CSVs behind cited numbers |
| `runs/` | 1042 per-run records including 208 `config.json` parameter sets |
| `tests/` | 134 tests |
| `scripts/` | entry points and manifest tooling |
| `docs/` | annotated tree, claim-to-result map, environment, checklist map |

`docs/PROJECT_TREE.md` annotates the full tree.
`docs/CLAIMS_TO_RESULTS.md` maps every reported number to its file and the
command that regenerates it.
`REPRODUCE.md` gives the command behind each figure and table.

## Conventions

Experiment drivers are Python modules and are run from the archive root:

```bash
python -m experiments.readout_basis.t1_abelian_convergence --help
```

All randomness flows through explicit seed arguments. Every benchmark run
writes a `config.json` recording its complete parameter set next to its
outputs, so any reported value is traceable to the run that produced it.



## Licence

See `LICENSE`. This archive is provided for peer review and for reproducing the
reported results.
