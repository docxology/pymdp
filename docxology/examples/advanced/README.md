# docxology/examples/advanced

> **Complex Active Inference Workloads**

This directory contains validated Python scenarios that exercise the advanced configuration and optimization features of `pymdp`. These scripts verify the library's ability to handle complex action dependencies, state-space optimization, and high-dimensional observation likelihoods.

## 🚀 Advanced Workloads

| Workload | Scenario Script | Description |
| :--- | :--- | :--- |
| **Complex Action Dependency** | `complex_action_dependency.py` | Validates agents where actions at $t$ depend on multi-factor states at $t-n$. |
| **Neural Encoder Integration** | `pymdp_with_neural_encoder.py` | Exercises the interface between discrete inference and continuous neural pre-processors. |
| **State Optimization** | `infer_states_optimization/` | Deep-dive benchmarks for FPI convergence speed and numerical stability. |

## 📊 Output Traces
Diagnostic artifacts, including action-probability heatmaps and optimized belief traces, are archived in:
`../../output/advanced/`

## 🤖 Agentic Contract
- **Memory Management**: When running neural encoder scenarios, ensure that the JAX device memory is properly cleared between epochs.
- **Factorization Rules**: Advanced models must adhere to the standardized factor-naming convention in `pymdp.utils`.

[Parent Examples Reference](../AGENTS.md)
