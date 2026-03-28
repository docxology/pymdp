# docxology/examples/inductive_inference — Structural Discovery Agentic Contract

## 0. Purpose
Machine-optimized index for the `inductive_inference` category. These scripts serve as the definitive execution targets for validating $I$-matrix (reachability) generation and structural discovery in Active Inference.

## 1. Inductive Registry

| Scenario | Primary API | Key Diagnostics |
| :--- | :--- | :--- |
| `reachability_demo.py` | `pymdp.control.generate_I_matrix` | `I_matrix`, `reachable_states`, `depth_trace` |

## 2. Invariant Constraints
Agents executing these scripts MUST enforce the following numerical invariants:
- **Reachability Range**: All values in the generated $I$ matrix must be within the set $[0, 1]$, representing the probability of goal achievement.
- **Threshold Pruning**: At the specified `threshold`, the number of states with $I(s) > 0$ must be strictly less than or equal to the total state-space cardinality.
- **Symmetry (Optional)**: For undirected state-transition graphs, the $I$ matrix at $d=1$ must exhibit symmetry where $I(s_i, s_j) > 0$.

## 3. Output Routing
All byproducts (NPZ, JSON, PNG) are routed via the [thin orchestrator](../../pkg/support/mirror_dispatch.py) to:
`docxology/output/inductive_inference/{scenario_stem}/*`

[Parent Examples Index](../AGENTS.md)
