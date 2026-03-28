# docxology/examples/legacy

> **NumPy-Era Compatibility & Regression Benchmarks**

This directory contains validated Python scenarios from the original (v1.0.0 — v1.50.0) `pymdp` codebase. These scripts verify that the modern JAX-backed engine maintains numerical parity with the legacy NumPy implementations and that core Active Inference loops remain functional for existing user models.

## 🚀 Legacy Workloads

| Workload | Scenario Script | Description |
| :--- | :--- | :--- |
| **GridWorld Legacy** | `grid_world_legacy.py` | Original navigation benchmark using `numpy` primitives. |
| **T-Maze Legacy** | `t_maze_legacy.py` | Original curiosity benchmark with legacy policy inference dispatch. |
| **BIF (Biological Inference)** | `bif_demo.py` | Original biological inference scenario for neural-like message passing. |

## 📊 Output Traces
Diagnostic artifacts, including parity-check logs and legacy belief trajectories, are archived in:
`../../output/legacy/`

## 🤖 Agentic Contract
- **Parity Tolerance**: Legacy benchmarks must match modern JAX outputs within $\epsilon=1e-4$. Discrepancies beyond this range are logged as regression failures.
- **Dispatch Rules**: These scenarios often bypass the `Agent` class in favor of direct `inference.update_posterior_states` calls; verify that these lower-level endpoints are still supported.

[Parent Examples Reference](../AGENTS.md)
