# docxology/examples/inference_and_learning

> **State Inference & Learning Comparative Benchmarks**

This directory contains validated Python scenarios that exercise the competitive and cooperative dynamics between different state inference and learning algorithms in `pymdp`. These scripts verify the numerical parity and convergence profiles of Fixed-Point Iteration (FPI), Marginal Message Passing (MMP), and Variational Message Passing (VMP).

## 🚀 Inference & Learning Workloads

| Workload | Scenario Script | Description |
| :--- | :--- | :--- |
| **Algorithm Audit** | `state_inference_learning_demo.py` | Compares FPI, MMP, and VMP accuracy over increasing observation noise. |

## 📊 Output Traces
Diagnostic artifacts, including VFE convergence trajectories and Dirichlet count heatmaps, are archived in:
`../../output/inference_and_learning/`

## 🤖 Agentic Contract
- **Convergence Parity**: Verify that all three inference algorithms converge to the same posterior $q(s)$ within $\epsilon=1e-5$ for simple 1-factor generative models.
- **Learning Rate Sensitivity**: Audit the stability of the Dirichlet update when the learning rate $\eta$ is varied across three orders of magnitude ($10^{-1}$ to $10^{-3}$).

[Parent Examples Reference](../AGENTS.md)
