# docxology/scripts — Tool Orchestration Agentic Contract

## 0. Purpose
Machine-optimized index for the `scripts` hierarchy. This directory contains the command-line entrypoints and subprocess wrappers for the docxology validation ecosystem.

## 1. Tool Registry

| Script | Responsibility | Execution Context |
| :--- | :--- | :--- |
| **[`run_docxology_orchestrations.py`](run_docxology_orchestrations.py)** | Categorical scenario execution. | `cwd = docxology/` |
| **[`run_docxology_notebooks.py`](run_docxology_notebooks.py)** | `nbval` testing for upstream tutorials. | `cwd = repo root` |
| **[`run_upstream_test_suite.sh`](run_upstream_test_suite.sh)** | Shell wrapper for `pytest test/`. | `cwd = repo root` |

## 2. Invariant Constraints
Agents invoking these scripts MUST enforce the following operational invariants:
- **Subprocess Isolation**: Every scenario must be executed in a fresh subprocess to prevent state leakage between simulation runs.
- **Manifest Adherence**: Only targets registered in `docxology/manifests/` may be executed as valid workloads.
- **Path Parity**: Scripts must automatically resolve `_DOCXOLOGY` and `_REPO` absolute paths to ensure portability across standardized and sidecar-only environments.

## 3. Output Routing
Subprocess logs and serialization commands from these scripts route data to:
`docxology/output/{category}/{scenario_stem}/*`

[Parent Sidecar Reference](../AGENTS.md)
