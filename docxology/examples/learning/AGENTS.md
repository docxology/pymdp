# docxology/examples/learning — Dirichlet Learning Agentic Contract

## 0. Purpose
Machine-optimized index for the `learning` category. These scripts serve as the definitive execution targets for validating parameter learning and concentration dynamics in `pymdp`.

## 1. Learning Registry

| Scenario | Primary API | Key Diagnostics |
| :--- | :--- | :--- |
| `obs_likelihood_learning.py` | `pymdp.learning.update_pA` | `pA_counts`, `posterior_H` |
| `transition_learning.py` | `pymdp.learning.update_pB` | `pB_counts`, `discovery_rate` |

## 2. Invariant Constraints
Agents executing these scripts MUST enforce the following numerical invariants:
- **Count Additivity**: The sum of Dirichlet concentration parameters in $pA$ must increase by exactly $1.0$ per valid observation-state pair (assuming $\eta=1.0$).
- **Probability Consistency**: The normalized likelihood $A$ derived from $pA$ must always represent a valid conditional distribution (columns sum to $1.0$).
- **Entropy reduction**: The Shannon entropy of the likelihood distribution $A$ should generally decrease as learning progresses (assuming a stationary environment).

## 3. Output Routing
All byproducts (NPZ, JSON, PNG) are routed via the [thin orchestrator](../../pkg/support/mirror_dispatch.py) to:
`docxology/output/learning/{scenario_stem}/*`

[Parent Examples Index](../AGENTS.md)
