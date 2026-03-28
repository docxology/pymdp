# docxology/examples/model_fitting — Parameter Recovery Agentic Contract

## 0. Purpose
Machine-optimized index for the `model_fitting` category. These scripts serve as the definitive execution targets for validating behavioral model inversion and state-space parameter recovery.

## 1. Registry

| Scenario | Primary API | Key Diagnostics |
| :--- | :--- | :--- |
| `svi_parameter_recovery.py` | `pymdp.model_fitting` | `recovered_params`, `SVI_loss`, `true_params` |
| `gridsearch_fitting.py` | `pymdp.model_fitting` | `likelihood_grid`, `MLE_estimate` |

## 2. Invariant Constraints
Agents executing these scripts MUST enforce the following numerical invariants:
- **Recovery Accuracy**: All recovered parameters must fall within the $95\%$ credible interval of the true generative parameters for simulated datasets of $N \ge 200$ trials.
- **Log-Likelihood Monotonicity**: During SVI fitting, the evidence lower bound (ELBO) must generally increase across training epochs.
- **Type Consistency**: Numerical outputs must be strictly cast to `float64` to maintain precision during the inversion of small-magnitude probability values.

## 3. Output Routing
All byproducts (NPZ, JSON, PNG) are routed via the [thin orchestrator](../../pkg/support/mirror_dispatch.py) to:
`docxology/output/model_fitting/{scenario_stem}/*`

[Parent Examples Index](../AGENTS.md)
