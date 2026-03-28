# docxology/pkg

> **Internal Support Infrastructure**

This folder houses the private Python package used by the docxology mirror scripts. It provides the heavy-lifting logic for orchestration, data extraction, and high-fidelity visualization.

## 📦 Module Breakdown

| Module | Purpose |
| :--- | :--- |
| **[`support/bootstrap.py`](support/bootstrap.py)** | Global path resolution and `sys.path` injection for the sidecar ecosystem. |
| **[`support/mirror_dispatch.py`](support/mirror_dispatch.py)** | The main registry of handlers that dispatch execution to real `pymdp` methods. |
| **[`support/viz.py`](support/viz.py)** | Headless-safe, premium visualization library (Matplotlib-based). |
| **[`support/analysis.py`](support/analysis.py)** | Thermodynamic metric derivation (Entropy, KL, Free Energy). |
| **[`support/si_fixtures.py`](support/si_fixtures.py)** | Pre-computed model structures for Sophisticated Inference benchmarks. |

## 🛠️ Installation Context

This package is designed to be installed as part of the docxology standalone environment. It pulls in `inferactively-pymdp` as an editable dependency from the parent directory:

```bash
# From docxology/
uv sync --group test
```

See [AGENTS.md](AGENTS.md) for the machine-readable contract.
