# docxology/examples/api — API Lifecycle Agentic Contract

## 0. Purpose
Machine-optimized index for the `api` category. These scripts serve as the definitive execution targets for validating core `pymdp` class constructors and initialization logic.

## 1. Lifecycle Registry

| Scenario | Primary API | Key Diagnostics |
| :--- | :--- | :--- |
| `model_construction_tutorial.py` | `pymdp.agent.Agent` | `A`, `B`, `C`, `D` matrices |

## 2. Invariant Constraints
Agents executing these scripts MUST enforce the following numerical invariants:
- **Matrix Shape Parity**: The $A$ and $D$ matrices must share the same observation factor dimension.
- **Initialization Stochasticity**: All resulting model matrices must pass the `_verify_invariants` stochasticity audit (summing to $1.0$ per column/row as appropriate).
- **Type Compliance**: All tensors must be verified as `jax.Array` or `np.ndarray` before serialization to NPZ.

## 3. Output Routing
All byproducts (NPZ, JSON, PNG) are routed via the [thin orchestrator](../../pkg/support/mirror_dispatch.py) to:
`docxology/output/api/{scenario_stem}/*`

[Parent Examples Index](../AGENTS.md)
