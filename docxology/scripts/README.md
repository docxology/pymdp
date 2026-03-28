# docxology/scripts

> **Validation Entrypoints & CLI Wrappers**

This directory provides the command-line entrypoints for the docxology validation ecosystem. These scripts handle path resolution and subprocess orchestration for Python scenarios and Jupyter Notebooks.

## 🛠️ Validation Tools

| Tool | Purpose | Description |
| :--- | :--- | :--- |
| **[`run_docxology_orchestrations.py`](run_docxology_orchestrations.py)** | **Scenario Runner** | Executes the categorical Python scenarios defined in the orchestrations manifest. |
| **[`run_docxology_notebooks.py`](run_docxology_notebooks.py)** | **Notebook Driver** | Invokes `nbval` on upstream notebooks to ensure execution parity with core library changes. |
| **[`run_upstream_test_suite.sh`](run_upstream_test_suite.sh)** | **Upstream Bridge** | Utility script to run the standard `pymdp` PyTest suite from the repository root. |

### 🚀 Usage Example
```bash
# Execute categorical Python scenarios
uv run python scripts/run_docxology_orchestrations.py

# Run notebook validation with strict output checking
uv run python scripts/run_docxology_notebooks.py --strict-output
```

> [!TIP]
> Use these scripts to verify that your local changes to `pymdp` do not break the diagnostic output or thermodynamic invariants of the examples.

See [AGENTS.md](AGENTS.md) for technical path resolution details and machine-readable contracts.
