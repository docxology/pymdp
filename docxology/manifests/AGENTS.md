# docxology/manifests — Registration Agentic Contract

## 0. Purpose
Machine-optimized index for the `manifests` hierarchy. This directory contains the plain-text registration lists that define the "ground truth" execution boundary for the docxology sidecar.

## 1. Hierarchy Registry

| Manifest | Execution Root | Deployment Strategy |
| :--- | :--- | :--- |
| **[`orchestrations.txt`](orchestrations.txt)** | `docxology/` | Primary categorical scenarios for `run_all.py`. |
| **[`notebooks_ci.txt`](notebooks_ci.txt)** | Repo Root | Essential `nbval` targets for pull requests. |
| **[`notebooks_nightly.txt`](notebooks_nightly.txt)** | Repo Root | Resource-intensive tutorials for nightly audits. |
| **[`python_scripts.txt`](python_scripts.txt)** | Repo Root | Upstream examples verified against current environment. |

## 2. Invariant Constraints
Agents interacting with these manifests MUST adhere to the following rules:
- **Path Resolvability**: Every non-commented path listed in a manifest must be resolvable from its specified `Execution Root`. This is enforced by `tests/test_manifests.py`.
- **Registration Only**: Do not pass arbitrary paths to orchestrator scripts; all execution targets must be formally registered here to ensure reproducible validation.
- **Line Discipline**: Manifests follow a strict "one path per line" format. Lines starting with `#` are ignored as comments.

## 3. Deployment Notes
Changes to these manifests directly alter the CI/CD test surface. Ensure all new scenarios added to `examples/` are registered in `orchestrations.txt` to enable automated validation.

[Parent Sidecar Reference](../AGENTS.md)
