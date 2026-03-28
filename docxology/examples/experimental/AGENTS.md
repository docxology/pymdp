# docxology/examples/experimental — Experimental Planning Agentic Contract

## 0. Purpose
Machine-optimized index for the `experimental` category. These scripts serve as the definitive execution targets for validating high-depth tree search and Monte Carlo inference in Active Inference.

## 1. Planning Registry

| Scenario | Primary API | Key Diagnostics |
| :--- | :--- | :--- |
| `sophisticated_inference/` | `pymdp.planning.si` | `pi_tree`, `recursive_G`, `node_count` |
| `mcts_*` targets | `pymdp.planning.mcts` | `rollout_vfe`, `visit_counts`, `depth` |

## 2. Invariant Constraints
Agents executing these scripts MUST enforce the following numerical invariants:
- **Tree Completeness**: Recursive SI search must return a policy tree where every branch terminates at the specified `search_horizon`.
- **EFE Monotonicity**: The expected free energy $G$ calculated via SI must be consistent with the single-horizon $G$ when depth $d=1$.
- **MCTS Convergence**: The policy posterior $q(\pi)$ derived from MCTS simulations must converge to the Boltzmann distribution of the estimated $G$ as $N \to \infty$.

## 3. Output Routing
All byproducts (NPZ, JSON, PNG) are routed via the [thin orchestrator](../../pkg/support/mirror_dispatch.py) to:
`docxology/output/experimental/{scenario_stem}/*`

[Parent Examples Index](../AGENTS.md)
