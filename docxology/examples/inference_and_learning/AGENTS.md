# docxology/examples/inference_and_learning — Algorithm Audit Agentic Contract

## 0. Purpose
Machine-optimized index for the `inference_and_learning` category. These scripts serve as the definitive execution targets for validating the numerical convergence and learning rate sensitivity of FPI, MMP, and VMP.

## 1. Algorithm Registry

| Scenario | Primary API | Key Diagnostics |
| :--- | :--- | :--- |
| `state_inference_learning_demo.py` | `pymdp.algos.*` | `qs_error`, `vfe_convergence`, `pA_update` |

## 2. Invariant Constraints
Agents executing these scripts MUST enforce the following numerical invariants:
- **Convergence Parity**: For the same generative model and observation, FPI and MMP must yield the same posterior $q(s)$ within $\epsilon=1e-5$.
- **VFE Minimization**: The Variational Free Energy (VFE) must be strictly non-increasing across iterations of the inference algorithm (assuming flat priors).
- **Learning Stability**: Dirichlet updates must not result in negative concentration parameters, even under extreme learning rates.

## 3. Output Routing
All byproducts (NPZ, JSON, PNG) are routed via the [thin orchestrator](../../pkg/support/mirror_dispatch.py) to:
`docxology/output/inference_and_learning/{scenario_stem}/*`

[Parent Examples Index](../AGENTS.md)
