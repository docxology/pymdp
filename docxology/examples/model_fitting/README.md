# docxology/examples/model_fitting

> **Parameter Estimation & Recovery Benchmarks**

This directory contains validated Python scenarios that exercise the model-inversion and parameter-recovery capabilities of `pymdp`. These scripts verify the ability to estimate hidden agent parameters (e.g., prior precision, learning rates) from observed behavior using variational and likelihood-based fitting techniques.

## 🚀 Model Fitting Workloads

| Workload | Scenario Script | Description |
| :--- | :--- | :--- |
| **SVI Recovery** | `svi_parameter_recovery.py` | Validates Stochastic Variational Inference (SVI) for recovering agent priors. |
| **GridSearch Fitting** | `gridsearch_fitting.py` | Exercises brute-force likelihood maximization over discrete parameter grids. |

## 📊 Output Traces
Diagnostic artifacts, including parameter-recovery plots and log-likelihood landscapes, are archived in:
`../../output/model_fitting/`

## 🤖 Agentic Contract
- **Recovery Accuracy**: Model fitting benchmarks must assert that recovered parameters are within $1\sigma$ of the true generative parameters for $N \ge 100$ trials.
- **Library Integration**: Verify that fitting procedures correctly interface with the `pybefit` library as a downstream dependency.

[Parent Examples Reference](../AGENTS.md)
