# docxology/examples/legacy — Regression Agentic Contract

## 0. Purpose
Machine-optimized index for the `legacy` category. These scripts serve as the definitive execution targets for validating numerical parity between JAX-backed and original NumPy-only implementations of Active Inference.

## 1. Registry

| Scenario | Primary API | Key Diagnostics |
| :--- | :--- | :--- |
| `grid_world_legacy.py` | `inference.update_posterior` | `qs_parity`, `vfe_parity` |
| `t_maze_legacy.py` | `control.update_policy_posterior` | `q_pi_parity`, `efe_parity` |

## 2. Invariant Constraints
Agents executing these scripts MUST enforce the following numerical invariants:
- **Numerical Parity**: Modern outputs must match legacy NumPy results within a strict tolerance of $\epsilon=1e-4$. Any discrepancy larger than $\epsilon$ signifies a breaking change in the tensor-dispatch logic.
- **Dispatch Integrity**: Legacy code often uses `pymdp.maths` and `pymdp.inference` directly; verify that these lower-level functional endpoints are correctly aliased to the modern JAX implementations.
- **Non-Regression**: No existing policy logic in the legacy set should fail to select the optimal path (as defined by the reward/preference matrix $C$) under default precision.

## 3. Output Routing
All byproducts (NPZ, JSON, PNG) are routed via the [thin orchestrator](../../pkg/support/mirror_dispatch.py) to:
`docxology/output/legacy/{scenario_stem}/*`

[Parent Examples Index](../AGENTS.md)
