# docxology/examples/inductive_inference

> **Structural Discovery & Reachability Benchmarks**

This directory contains validated Python scenarios that exercise the structural inference capabilities of `pymdp`. These scripts verify the generation of inductive ($I$) matrices, which represent the reachability of goal states across arbitrary transition structures and policy depths.

## 🚀 Inductive Workloads

| Workload | Scenario Script | Description |
| :--- | :--- | :--- |
| **Reachability Analysis** | `reachability_demo.py` | Validates the recursive `generate_I_matrix` function against known-connectivity graphs. |

## 📊 Output Traces
Diagnostic artifacts, including color-coded reachability matrices and entropy maps over structural discovery, are archived in:
`../../output/inductive_inference/`

## 🤖 Agentic Contract
- **Threshold Integrity**: Verify that the `threshold` parameter in `generate_I_matrix` correctly prunes unreachable states from the marginalized policy space.
- **Graph Consistency**: Inductive benchmarks must use the `si_fixtures.py` graph definitions to ensure parity with the experimental planning suite.

[Parent Examples Reference](../AGENTS.md)
