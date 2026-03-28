# docxology/examples

> **Active Inference Workload Taxonomy (AGENTS.md)**

This directory houses the 32 core execution targets used for sidecar validation. Every scenario is a **machine-verifiable workload** that exercises specific functional subsets of the `pymdp` library.

## 🛠️ Workload Matrix

Agents delegating tasks to this directory must adhere to the following categorical boundaries:

| Category | Agentic Responsibility | Mathematical Focus |
| :--- | :--- | :--- |
| **`advanced/`** | Complex dependency resolution. | Neural Encoders, Action-State coupling. |
| **`api/`** | Basic object lifecycle checks. | Agent instantiation, Parameter normalization. |
| **`docxology/`** | Sidecar reflection and rollouts. | Pipeline stability, Trace extraction. |
| **`envs/`** | Standard environment integration. | T-Maze, GridWorld, POMDP dynamics. |
| **`experimental/`** | Planning & sophistication audits. | Sophisticated Inference (SI), MCTS. |
| **`inductive_inference/`** | Structural state discovery. | $I$-matrix generation, Reachability. |
| **`inference_and_learning/`** | Algorithm comparative audits. | FPI vs. MMP vs. VMP state inference. |
| **`learning/`** | Bayesian parameter updates. | Dirichlet counts, $pA/pB$ learning rules. |
| **`model_fitting/`** | Parameter recovery and fitting. | Log-likelihood maximization, GridSearch. |
| **`sparse/`** | Optimization & sparsity audits. | Sparse tensor dot-products and factorizations. |
| **`legacy/`** | Regression & backward parity. | Original `pymdp` tutorial compatibility. |

## 🧩 Execution Contract

1.  **Zero-Mock Enforcement**: Every script MUST use real `pymdp` methods. Mocks are strictly prohibited within the `examples/` tree.
2.  **Configuration Injection**: Scenarios take an `OrchestrationConfig` object via `pkg.support.mirror_dispatch`.
3.  **Trace Minimization**: When `config.fast` is `true`, scripts must truncate loops to $\le 2$ timesteps to ensure rapid smoke-testing.

## 🤖 Machine Guidance

- **Registry**: All scenarios here MUST be registered in `docxology/manifests/orchestrations.txt` to be included in the `run_all.py` pipeline.
- **Handlers**: Every example sub-directory must map to a handler function in `pkg/support/mirror_dispatch.py`.

[Parent Sidecar Reference](../AGENTS.md)
