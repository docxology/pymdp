# docxology/examples/learning

> **Bayesian Parameter Learning Benchmarks**

This directory contains validated Python scenarios that exercise the Dirichlet-based learning rules in `pymdp`. These scripts verify the incremental updating of observation likelihoods ($pA$) and transition dynamics ($pB$) based on accumulating evidence (observations and inferred states).

## 🚀 Learning Workloads

| Workload | Scenario Script | Description |
| :--- | :--- | :--- |
| **Likelihood Learning** | `obs_likelihood_learning.py` | Validates the $pA$ update rule across multiple observation factors. |
| **Transition Learning** | `transition_learning.py` | Exercises the $pB$ update rule for discovering environmental dynamics. |

## 📊 Output Traces
Diagnostic artifacts, including Dirichlet count matrices and learning-curve entropy maps, are archived in:
`../../output/learning/`

## 🤖 Agentic Contract
- **Posterior Sensitivity**: Learning benchmarks must verify that the concentration parameters for the Dirichlet distributions remain strictly positive.
- **Factorization Rules**: When learning multi-factor models, ensure that the $pA$ and $pB$ shapes precisely match the generative model's factor structure.

[Parent Examples Reference](../AGENTS.md)
