# Environment

## Reference environment

Every number reported in the paper and supplementary document was produced on:

| | |
|---|---|
| Hardware | Apple M4, 10 CPU cores, 16 GB RAM |
| Operating system | macOS 26.5.2, arm64 |
| Interpreter | CPython 3.14.5 |
| Quantum hardware | IBM Heron processors `ibm_marrakesh` and `ibm_fez` |

No GPU is used. The library is pure NumPy and every simulated result is
deterministic given its seed.

## Pinned versions

`requirements-lock.txt` holds the exact versions:

| package | version |
|---|---|
| numpy | 2.5.1 |
| scipy | 1.18.0 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.0 |
| pandas | 3.0.3 |
| qiskit | 2.5.0 |
| qiskit-aer | 0.17.2 |
| qiskit-machine-learning | 0.9.0 |
| qiskit-ibm-runtime | 0.47.0 |
| pennylane | 0.45.1 |
| networkx | 3.6.1 |
| rustworkx | 0.18.0 |
| pytest | 8.4.2 |

`requirements.txt` gives supported ranges rather than exact pins. Python 3.11
through 3.14 are supported.

## Installing

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e .
```

The editable install puts `src/` on the import path, so `import cqfa` and
`import datasets` resolve. Experiment drivers, analysis code and figure scripts
are run as modules from the archive root, which puts the root on the path so
`experiments.*`, `analysis.*` and `figures.*` resolve as well.

## Optional components

The quantum stack is required only by the circuit-construction and hardware
experiments. Every simulated result, every figure and the full test suite run
without it.



## Determinism

All randomness flows through explicit seed arguments. Data seeds follow the
convention `1000 + seed` so the quantum and classical arms see the same
instances for a given seed. Each benchmark run writes a `config.json` beside
its outputs recording its complete parameter set.
