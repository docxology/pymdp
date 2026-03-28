# docxology/examples/api

> **Core Library Lifecycle Benchmarks**

This directory contains validated Python scenarios that exercise the foundational API surface of `pymdp`. These scripts verify the construction, normalization, and validation of the core generative model components ($A, B, C, D$ matrices) and the lifecycle of the `Agent` class.

## 🚀 API Workloads

| Workload | Scenario Script | Description |
| :--- | :--- | :--- |
| **Model Construction** | `model_construction_tutorial.py` | Validates the manual and automated assembly of multi-factor generative models. |

## 📊 Output Traces
Diagnostic artifacts, including visual representations of the likelihood and transition matrices, are archived in:
`../../output/api/`

## 🤖 Agentic Contract
- **Normalization Audits**: API benchmarks must trigger the `_verify_invariants` check to ensure that all manually constructed matrices are row/column-stochastic where required.
- **Factory Consistency**: Verify that `pymdp.utils.obj_to_obj_array` correctly handles heterogeneous factor dimensions during agent initialization.

[Parent Examples Reference](../AGENTS.md)
