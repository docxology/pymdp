# docxology/examples/experimental

> **Cutting-Edge Planning & Tree-Search Benchmarks**

This directory contains validated Python scenarios for the most computationally intensive and theoretically advanced features of `pymdp`. These scripts verify the sophisticated planning capabilities, including temporal tree-search, counterfactual reasoning, and Monte Carlo Tree Search (MCTS) integrations.

## 🚀 Experimental Workloads

| Workload | Scenario Category | Description |
| :--- | :--- | :--- |
| **Sophisticated Inference** | [`sophisticated_inference/`](sophisticated_inference/) | Validates the SI recursive policy search over T-Maze and GraphWorld. |
| **Tree-Search (MCTS)** | `mcts_*` targets | Exercises the Monte Carlo Tree Search dispatch for high-dimensional planning. |

## 📊 Output Traces
Diagnostic artifacts, including planning trees and Expected Free Energy (EFE) heatmaps over multiple horizons, are archived in:
`../../output/experimental/`

## 🤖 Agentic Contract
- **Resource Constraints**: These benchmarks are marked as "heavy" and require `MPLBACKEND=Agg` for headless execution of high-depth tree visualizations.
- **Invariance Rules**: Verify that the cumulative EFE for the optimal policy path in `si_graph_world.py` is monotonically non-increasing across depths.

[Parent Examples Reference](../AGENTS.md)
