# docxology/pkg/support — Thin Orchestrator Agentic Contract

## 0. Purpose
Machine-optimized index for the `support` package. This directory implements the core execution and validation logic of the docxology sidecar, delegating heavy lifting to real `pymdp` methods.

## 1. File Role Registry

| File | Agent Role & Responsibility | Technical Depth |
| :--- | :--- | :--- |
| **`mirror_dispatch.py`** | **Main Dispatcher**: Maps scenario strings to real `pymdp` execution loops. | High |
| **`viz.py`** | **Viz Engine**: Headless-safe plotting for beliefs, EFE heatmaps, and action probs. | High |
| **`analysis.py`** | **Math Lab**: Derives Shannon entropy $H(q)$ and verifies probability normalization. | Medium |
| **`bootstrap.py`** | **Environment**: Sets `Agg` backend, configures logging, and adjusts `sys.path`. | Low |
| **`checks.py`** | **Sanity**: Probes for optional dependencies (`torch`, `pybefit`) and JAX availability. | Low |
| **`si_fixtures.py`** | **Models**: Pre-defined $A, B, C, D$ tensors for T-Maze and Graph World benchmarks. | Medium |

## 2. Invariant Constraints
Agents executing these scripts MUST enforce the following numerical invariants:
- **Normalization Audit**: Every belief state $q(s)$ extracted must sum to $1.0 \pm 1e-3$ across all factors.
- **Headless Safety**: All Matplotlib invocations MUST use the `Agg` backend to avoid display-server errors in CI.
- **JSON Serialization**: All return dictionaries from `HANDLERS` must be strictly cast to native Python types via `_to_serializable` before storage.

## 3. Output Routing
All byproducts (NPZ, JSON, PNG) are routed via the [Global Sidecar Root](../../README.md) to:
`docxology/output/{category}/{scenario_stem}/*`

[Parent Package Reference](../AGENTS.md)
