# docxology/examples/advanced — Advanced Agentic Contract

## 0. Purpose
Machine-optimized index for the `advanced` category. These scripts serve as the definitive execution targets for validating $A/O/X$ dependencies and neural-symbolic interfaces in Active Inference.

## 1. Workload Registry

| Scenario | Primary API | Key Diagnostics |
| :--- | :--- | :--- |
| `complex_action_dependency.py` | `pymdp.control` | `action_selection_prob`, `dependency_trace` |
| `pymdp_with_neural_encoder.py` | `jax.numpy` | `latent_states`, `encoder_loss` |
| `infer_states_optimization/` | `pymdp.algos.fpi` | `convergence_rate`, `step_size` |

## 2. Invariant Constraints
Agents executing these scripts MUST enforce the following numerical invariants:
- **Factor Dependency**: In `complex_action_dependency.py`, the action marginals must be conditioned on the correct sub-factor of the state space.
- **Gradient Continuity**: For neural integrations, the JAX gradients must be traceable through the `qs` posterior update.
- **Floating-Point Stability**: All optimized belief updates must maintain `f32` precision without underflow in the softmax layer.

## 3. Output Routing
All byproducts (NPZ, JSON, PNG) are routed via the [thin orchestrator](../../pkg/support/mirror_dispatch.py) to:
`docxology/output/advanced/{scenario_stem}/*`

[Parent Examples Index](../AGENTS.md)
