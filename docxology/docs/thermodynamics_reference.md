# Thermodynamics Reference

Mathematical foundations for the Active Inference diagnostics computed by the docxology orchestrator. All formulas map directly to real pymdp function implementations.

---

## 1. Variational Free Energy (VFE)

**The quantity an agent minimizes during perception (state inference).**

### Definition

```
F[q] = E_q[log q(s)] − E_q[log p(o, s)]
```

### Decomposition: Accuracy − Complexity

```
F = D_KL[q(s) ‖ p(s)] − E_q[log p(o|s)]
  = Complexity − Accuracy
```

| Component | Formula | Interpretation | pymdp Implementation |
|---|---|---|---|
| **Accuracy** | `E_q[log p(o\|s)]` | How well beliefs explain current observations | `maths.compute_accuracy(qs, A, obs)` |
| **Complexity** | `D_KL[q(s) ‖ p(s)]` | How far beliefs deviate from prior expectations | KL term in `maths.calc_vfe()` |
| **Full VFE** | `Complexity − Accuracy` | Total surprise bound | `maths.calc_vfe(qs, prior, ll)` |

### Docxology Implementation

```python
# In pkg/support/analysis.py
def compute_accuracy_complexity(q_s, p_o_given_s, obs, prior):
    likelihood = jnp.clip(p_o_given_s[obs, :], 1e-12, 1.0)
    accuracy = jnp.sum(q_s * jnp.log(likelihood))

    q_safe = jnp.clip(q_s, 1e-12, 1.0)
    prior_safe = jnp.clip(prior, 1e-12, 1.0)
    complexity = jnp.sum(q_s * (jnp.log(q_safe) - jnp.log(prior_safe)))

    return accuracy, complexity
```

### Orchestrator Behavior

- If `vfe` or `F` key present in handler output with size > 1 → triggers `plot_free_energy` (tomato line plot)
- Terminal and mean VFE values reported in Performance Insights table of execution reports

---

## 2. Expected Free Energy (EFE)

**The quantity an agent minimizes during action selection (policy inference).**

### Definition

```
G(π) = Σ_τ E_q(o,s|π) [ log q(s|π) − log p(o, s) ]
```

### Decomposition: Epistemic + Pragmatic + Novelty

```
G(π) = Σ_τ [ −Epistemic(τ) − Pragmatic(τ) − Novelty(τ) ]
```

| Component | Formula | Interpretation | pymdp Implementation |
|---|---|---|---|
| **Epistemic** | `−E_q(o\|s,π)[ H[p(s\|o)] − H[q(s\|π)] ]` | Information gain — drives exploration | `control.compute_info_gain()` |
| **Pragmatic** | `−E_q(o\|π)[ log p(o\|C) ]` | Expected utility — drives goal-directed behavior | `control.compute_expected_utility()` |
| **Novelty** | `−E_q(s\|π)[ D_KL[q(A) ‖ p(A)] ]` | Parameter information gain — drives learning | `control.calc_negative_pA_info_gain()` |
| **Full −G(π)** | `Epistemic + Pragmatic + Novelty` | Negative expected free energy | `control.compute_neg_efe_policy()` |

### Policy Posterior

```
q(π) = σ(−G(π)) = softmax over negative EFE
```

The policy posterior assigns higher probability to policies with lower expected free energy — balancing exploration (epistemic gain) and exploitation (pragmatic value).

### Orchestrator Behavior

- `neg_efe` key with T≥2, π≥2 → triggers `plot_efe_trajectory` (best/mean −G) and `plot_neg_efe_heatmap` (−G landscape)
- `G_epistemic` + `G_pragmatic` → triggers `plot_efe_components` (grouped breakdown)
- Terminal neg_efe values reported in Performance Insights

---

## 3. Shannon Entropy

**Measures uncertainty in a probability distribution.**

### Definition

```
H(q) = −Σ q(s) log q(s)
```

| Property | Value |
|---|---|
| Minimum | 0 (certainty: all mass on one state) |
| Maximum | log(N) (uniform over N states) |
| Units | nats (natural logarithm) |

### pymdp Implementation

```python
# pymdp/maths.py
def stable_entropy(x):
    return -jnp.sum(stable_xlogx(x))
```

### Docxology Implementations

**In `pkg/support/analysis.py`:**
```python
def compute_entropy(probs, axis=-1):
    p_safe = jnp.clip(probs, 1e-12, 1.0)
    return -jnp.sum(probs * jnp.log(p_safe), axis=axis)
```

**Retroactive derivation in `mirror_dispatch.py`:**
```python
# Auto-computed when qs present but H_qs not
if "qs" in info and "H_qs" not in info:
    for q in qs_raw:
        q_safe = np.clip(q_arr, 1e-12, 1.0)
        h = -np.sum(q_safe * np.log(q_safe))
        h_seq.append(float(h))
    info["H_qs"] = h_seq
```

### Orchestrator Behavior

- Retroactively derived from `qs` belief sequences when not explicitly logged
- `qs` with ≥2 timesteps → triggers `plot_entropy_trajectory` (darkorange line with fill)
- Terminal and mean H(q) values reported in Performance Insights

---

## 4. KL Divergence

**Measures how much one distribution diverges from another.**

### Definition

```
D_KL(q ‖ p) = Σ q(s) log(q(s) / p(s))
            = Σ q(s) [log q(s) − log p(s)]
```

| Property | Value |
|---|---|
| Minimum | 0 (q = p) |
| Not symmetric | D_KL(q‖p) ≠ D_KL(p‖q) |
| Units | nats |

### Docxology Implementation

```python
# pkg/support/analysis.py
def trajectory_divergence(qs_seq, prior):
    for qs in qs_seq:
        kl = jnp.sum(qs * (jnp.log(q_safe) - jnp.log(p_safe)))
        divergences.append(kl)
    return jnp.array(divergences)
```

### Orchestrator Behavior

- `qs` + `D_matrix` both present → triggers `plot_kl_divergence_trajectory` (crimson line)
- `KL` key → reported in Performance Insights if present

---

## 5. Dirichlet Learning

**Bayesian parameter learning that updates the generative model from experience.**

### Update Rules

**Observation likelihood:**
```
pA_m(o, s) ← pA_m(o, s) + η · o_m(o) ⊗ q(s)
```

*"I saw observation o when I believed state s — strengthen that association."*

**Transition dynamics:**
```
pB_f(s', s, a) ← pB_f(s', s, a) + η · q(s') ⊗ q(s)
```

*"State s' followed state s under action a — strengthen that transition."*

### Expected Parameters

```
E[A] = pA / Σ_o pA    (normalized Dirichlet)
E[B] = pB / Σ_s' pB
```

### pymdp Implementation

| Function | Description |
|---|---|
| `learning.update_obs_likelihood_dirichlet(pA, A_deps, qs, obs, lr)` | Update pA from single observation |
| `learning.update_obs_likelihood_dirichlet_m(...)` | Per-modality variant |
| `learning.update_state_transition_dirichlet(pB, B_deps, qs_prev, qs_curr, action, lr)` | Update pB from single transition |
| `learning.update_state_transition_dirichlet_f(...)` | Per-factor variant |

### Orchestrator Behavior

- `qA_delta` in handler output → logged in diagnostics
- Learning rate `lr` logged in diagnostics

---

## 6. Inductive Inference (Reachability)

**Biases policy selection toward states from which goals are reachable.**

### The I Matrix

```
I(s) = Σ_d P(reach goal | start at s, depth=d) > threshold
```

The reachability matrix encodes which states can lead to goal states within a given planning horizon.

### Policy Bias

```
G_inductive(π) = G(π) + γ · I · q(s|π)
```

Policies that lead to high-reachability states receive a bonus in their EFE evaluation.

### pymdp Implementation

```python
control.generate_I_matrix(H, B, threshold, depth)
```

| Parameter | Default | Effect |
|---|---|---|
| `H` | — | Goal state indicator vector |
| `B` | — | Transition matrix |
| `threshold` | 0.5 | Minimum reachability probability |
| `depth` | 4 | Planning horizon depth |

### Orchestrator Behavior

- `I_matrix` or `I` with ndim ≥ 2 → triggers `plot_reachability_matrix` (YlOrRd heatmap)
- I matrix natively archived in `_model_trace.npz`

---

## 7. Sophisticated Inference

**Replaces flat policy enumeration with lookahead tree search.**

### Algorithm

```
For each possible action sequence:
  1. Predict future observations: q(o|s,a) = A · B(a) · q(s)
  2. Update beliefs: q(s|o) via Bayesian update
  3. Evaluate −G at each tree node
  4. Prune low-probability branches
  5. Backpropagate values to root
```

### pymdp Implementation

```python
planning.si.si_policy_search(
    horizon, max_nodes=512, max_branching=32,
    policy_prune_threshold=0.01, entropy_stop_threshold=0.1
)
```

### MCTS Alternative

```python
planning.mcts.mcts_policy_search(
    max_depth=4, num_simulations=128
)
```

Uses `mctx.gumbel_muzero_policy` with a custom `recurrent_fn` implementing Active Inference dynamics.

---

## Mathematical Invariants

The orchestrator enforces these properties on every run:

| Property | Check | Tolerance |
|---|---|---|
| Beliefs are valid distributions | `Σ_s q(s) = 1` | ±1e-3 |
| Policy posterior is valid | `Σ_π q(π) = 1` | ±1e-3 |
| Likelihood is column-stochastic | `Σ_o A(o,s) = 1` | ±1e-3 |
| Transitions are column-stochastic | `Σ_s' B(s',s,a) = 1` | ±1e-3 |

Violations are logged but do not crash the pipeline — facilitating debugging of experimental agents.

---

## See Also

- [AGENTS.md](AGENTS.md) — formal API tables for all referenced functions
- [orchestrator_internals.md](orchestrator_internals.md) — how these metrics are computed in the pipeline
- [visualization_reference.md](visualization_reference.md) — plot functions that visualize these quantities
