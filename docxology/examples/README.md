# docxology/examples

> **Active Inference Validation Workloads**

This directory contains the 32 core execution targets that drive the docxology validation ecosystem. These scripts are not mere demonstrations; they are **mathematically authoritative benchmarks** that exercise the real `pymdp` API surface.

## 📂 Workload Taxonomy

The examples are organized into 11 categories covering the entire breadth of Active Inference operations:

| Category | Responsibility |
| :--- | :--- |
| **[`advanced/`](advanced/)** | Complex action-dependencies and neural-encoder mappings. |
| **[`api/`](api/)** | Basic `pymdp` object construction and API smoke tests. |
| **[`docxology/`](docxology/)** | Internal sidecar-specific benchmarks and rollout validations. |
| **[`envs/`](envs/)** | Standard Active Inference environments (T-Maze, GridWorld, etc.). |
| **[`experimental/`](experimental/)** | Cutting-edge Sophisticated Inference (SI) and MCTS benchmarks. |
| **[`inductive_inference/`](inductive_inference/)** | Reachability analysis and inductive state discovery targets. |
| **[`inference_and_learning/`](inference_and_learning/)** | Competitive benchmarks for FPI, MMP, and VMP algorithms. |
| **[`learning/`](learning/)** | Dirichlet parameter learning and observation likelihood updates. |
| **[`model_fitting/`](model_fitting/)** | Likelihood-based model inversion and parameter estimation. |
| **[`sparse/`](sparse/)** | Numerical parity checks between dense and sparse tensor backends. |
| **[`legacy/`](legacy/)** | Regression tests against the original `pymdp` tutorial series. |

## 🚀 Batch Execution
To run every validated scenario in this tree and generate a full set of [diagnostic artifacts](../output/):

```bash
cd docxology
uv run python run_all.py
```

> [!TIP]
> For a deep-dive into the specific mathematical properties of every example, refer to the [Operator Index](../docs/examples_catalog.md).

See [AGENTS.md](AGENTS.md) for the machine-readable contract.
