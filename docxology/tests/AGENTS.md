# docxology/tests — Structural Validation Agentic Contract

## 0. Purpose
Machine-optimized index for the `tests` hierarchy. This directory contains the `pytest` suite for validating the integrity of the docxology orchestrator and manifest resolution.

## 1. Test Registry

| Component | Responsibility | Technical Depth |
| :--- | :--- | :--- |
| **[`conftest.py`](conftest.py)** | Global path resolution and mock-free agent factories. | Low |
| **[`test_manifests.py`](test_manifests.py)** | Verifies resolvability of all registry paths. | Medium |
| **[`test_orchestrations.py`](test_orchestrations.py)** | Smoke tests every scenario in `--fast` mode. | High |
| **[`test_notebook_runner_smoke.py`](test_notebook_runner_smoke.py)** | Verifies CLI arguments and help-output of drivers. | Medium |

## 2. Invariant Constraints
Agents executing these tests MUST enforce the following operational invariants:
- **Path Purity**: Tests must use absolute path resolution via `Path(__file__).resolve()` to ensure stability across heterogeneous CI environments.
- **Independence**: Test logic must not depend on the volatile state of `docxology/output/` or pre-existing build artifacts.
- **Zero-Mock Policy**: Structural validation must prioritize real `pymdp` method execution over mocks to ensure documentation-to-code parity.

## 3. Deployment Notes
This suite is executed via `uv run pytest tests/ -v`. It is the primary gatekeeper for ensuring that new examples and manifests are correctly registered and resolvable.

[Parent Sidecar Reference](../AGENTS.md)
