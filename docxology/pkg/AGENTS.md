# docxology/pkg — Infrastructure Agentic Contract

## 0. Purpose
Machine-optimized index for the `pkg` support hierarchy. This directory contains the private utility libraries for orchestrating Active Inference validations and thermodynamic analytics.

## 1. Component Registry

| Component | Responsibility | Technical Depth |
| :--- | :--- | :--- |
| **[`support/`](support/AGENTS.md)** | Core helpers for orchestration and visualization. | High |
| **`__init__.py`** | Package resolution and namespace isolation. | Low |

## 2. Invariant Constraints
Agents interacting with this infrastructure MUST adhere to the following rules:
- **Namespace Isolation**: Do not import `pymdp` directly within `pkg/` unless wrapped in the `mirror_dispatch` or `bootstrap` logic.
- **Path Robustness**: All internal path resolutions MUST use `Path(__file__).resolve()` to ensure stability across variable `CWD` execution contexts.
- **Trace Continuity**: Orchestration utilities must verify that the `qs` belief states are returned as a continuous list matching the simulation timesteps $T$.

## 3. Deployment Notes
This package is intended for internal use by `run_all.py` and the `examples/` suite. It is not a public library and does not follow Semantic Versioning.

[Parent Sidecar Reference](../AGENTS.md)
