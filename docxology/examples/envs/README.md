# docxology/examples/envs

> **Active Inference Environment Benchmarks**

This directory contains the validated Python scenarios for core Active Inference environments. These scripts exercise the `pymdp.envs` module, ensuring that the POMDP dynamics, observation likelihoods, and transition structures are correctly implemented and preserved across library updates.

## 🏝️ Environment Matrix

Every scenario in this folder is a verified execution target that is automatically processed by the [thin orchestrator](../../pkg/support/mirror_dispatch.py).

| Environment | Scenario Script | Description |
| :--- | :--- | :--- |
| **GridWorld** | `grid_world_demo.py` | Validates spatial navigation and policy inference in 2D discrete grids. |
| **T-Maze** | `t_maze_classic.py` | The foundational curiosity benchmark testing epistemic information gain. |
| **GraphWorld** | `graph_navigation.py` | Generalizes navigation to non-Euclidean state spaces with arbitrary connectivity. |
| **Rollout** | `minimal_rollout.py` | Exercises the `pymdp.envs.Rollout` wrapper for temporal planning. |

## 📊 Output Traces
Diagnostic artifacts, including belief heatmaps and belief trajectories for these environments, are archived in:
`../../output/envs/`

## 🤖 Agentic Contract
- **State Space Validation**: Agents must ensure that the $A$ and $B$ matrices provided to these environments remain properly normalized.
- **Rollout Consistency**: Use `minimal_rollout.py` to verify that the environment's `step()` function correctly advances both the hidden state and the generative prior.

[Parent Examples Reference](../AGENTS.md)
