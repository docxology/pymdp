# docxology/examples/envs — Environment Agentic Contract

## 0. Purpose
Machine-optimized index for the `envs` category. These scripts serve as the definitive execution targets for validating $A/B/C/D$ dynamics in Active Inference environments.

## 1. Environment Registry

| Scenario | Primary API | Key Diagnostics |
| :--- | :--- | :--- |
| `grid_world_demo.py` | `pymdp.envs.GridWorld` | `qs`, `action`, `reward` |
| `t_maze_classic.py` | `pymdp.envs.TMaze` | `EFE`, `epistemic_gain` |
| `graph_navigation.py` | `pymdp.maths.factor_dot` | `transition_accuracy` |
| `minimal_rollout.py` | `pymdp.envs.Rollout` | `trajectory_VFE` |

## 2. Invariant Constraints
Agents executing these scripts MUST enforce the following numerical invariants:
- **Transition Stochasticity**: Columns of the environmental transition matrix $B$ must sum to $1.0$.
- **Belief Convergence**: Posterior beliefs $q(s)$ must be normalized via `pymdp.maths.softmax` or equivalent.
- **Trace Continuity**: Action sequences must exhibit zero gaps across the specified $T$ timesteps.

## 3. Output Routing
All byproducts (NPZ, JSON, PNG) are routed via the [thin orchestrator](../../pkg/support/mirror_dispatch.py) to:
`docxology/output/envs/{scenario_stem}/*`

[Parent Examples Index](../AGENTS.md)
