# docxology/tests

> **Sidecar Structural & Functional Validation**

This directory contains the `pytest` suite for the docxology sidecar. These tests ensure the integrity of the orchestrator, manifest resolution, and simulation consistency.

## 🧪 Test Suite Breakdown

| File | Role | Description |
| :--- | :--- | :--- |
| **[`test_manifests.py`](test_manifests.py)** | **Structural** | Verifies that all paths in `manifests/*.txt` are valid and resolvable. |
| **[`test_orchestrations.py`](test_orchestrations.py)** | **Functional** | Smoke tests every Python scenario using the `--fast` flag. |
| **[`test_notebook_runner_smoke.py`](test_notebook_runner_smoke.py)** | **Interface** | Verifies CLI argument handling for the notebook driver. |
| **[`conftest.py`](conftest.py)** | **Infrastructure** | Provides shared fixtures and global path resolution logic. |

### 🚀 Running Tests
```bash
# Standard fast checks
uv run pytest tests/ -v

# Include slow notebook validation
uv run pytest tests/ -m slow
```

> [!IMPORTANT]
> These tests are confined to the `docxology/` directory and do not execute the main `pymdp` unit tests. To run the full upstream suite, use `scripts/run_upstream_test_suite.sh`.

See [AGENTS.md](AGENTS.md) for technical markers and environment variable details.
