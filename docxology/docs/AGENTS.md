# docxology/docs — Technical Reference

## Purpose

Comprehensive technical reference tying the **pymdp** Active Inference package, the **docxology** validation sidecar, and all supporting infrastructure together. Indexes **32 runnable examples**, **upstream tests**, **published documentation**, and per-timestep diagnostic output.

---

## Part I: pymdp Package Architecture

### Package Layout

```
pymdp/
├── __init__.py              # Package entry point
├── agent.py                 # Agent class (equinox Module) — the central abstraction
├── algos.py                 # 34 inference algorithm implementations (FPI, MMP, VMP, HMM)
├── control.py               # 18 policy inference functions (EFE, inductive, sampling)
├── inference.py             # 15 state inference functions (posterior update, smoothing)
├── learning.py              # 4 Dirichlet parameter update functions
├── maths.py                 # 32 mathematical primitives (factor_dot, entropy, KL, VFE)
├── utils.py                 # 31 array construction/normalization utilities
├── distribution.py          # 5 distribution helper classes
├── likelihoods.py           # 2 likelihood computation wrappers
├── envs/                    # Environment implementations
│   ├── env.py               # PymdpEnv base class + cat_sample
│   ├── tmaze.py             # TMaze, SimplifiedTMaze (BaseTMaze)
│   ├── generalized_tmaze.py # GeneralizedTMazeEnv + maze parser
│   ├── cue_chaining.py      # CueChainingEnv (epistemic chaining)
│   ├── graph_worlds.py      # GraphEnv (cluster-based graphs)
│   ├── grid_world.py        # GridWorld (N×M navigable grid)
│   └── rollout.py           # rollout() engine + online learning
├── planning/                # Advanced planning algorithms
│   ├── si.py                # Sophisticated Inference (tree search)
│   ├── mcts.py              # MCTS via mctx library
│   └── visualize.py         # Plan tree visualization
└── legacy/                  # NumPy-era backward compatibility layer
    ├── agent.py             # Legacy Agent class
    ├── algos/               # Legacy FPI, MMP implementations
    ├── envs/                # Legacy TMazeEnv, GridWorldEnv
    └── ...                  # maths, utils, control, inference, learning
```

### Core: `agent.py` — The Agent Class

```python
class Agent(eqx.Module):
    """Active Inference agent as an Equinox module (JAX-compatible, JIT-safe)."""
```

**Generative model parameters:**

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `A` | `list[Array]` | `[num_obs_m × Π(num_states_f)]` per modality | Likelihood mapping p(o\|s) |
| `B` | `list[Array]` | `[num_states_f × num_states_f × num_controls_f]` per factor | Transition dynamics p(s'\|s,a) |
| `C` | `list[Array]` | `[num_obs_m]` per modality | Prior preferences (log-scale) |
| `D` | `list[Array]` | `[num_states_f]` per factor | Initial state prior p(s₀) |
| `pA` | `list[Array]` or `None` | Same as A | Dirichlet prior for A learning |
| `pB` | `list[Array]` or `None` | Same as B | Dirichlet prior for B learning |

**Dependency specifications:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `A_dependencies` | `list[list[int]]` | Which state factors each modality depends on |
| `B_dependencies` | `list[list[int]]` | Which state factors each factor's transitions depend on |
| `B_action_dependencies` | `list[list[int]]` | Which control factors each factor's action depends on |

**Key methods:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `infer_states` | `(obs, empirical_prior, return_info) → (qs, info)` | Posterior belief update q(s) |
| `infer_policies` | `(qs) → (q_pi, neg_efe)` | Policy posterior q(π) via expected free energy |
| `sample_action` | `(q_pi, rng_key) → action` | Sample action from policy posterior |
| `update_empirical_prior` | `(action, qs) → empirical_prior` | Compute p(s'\|a) for next step |

**Active Inference cycle:**
```
obs → infer_states(obs) → qs → infer_policies(qs) → q_pi → sample_action(q_pi) → action
                                    ↑                                                   ↓
                                update_empirical_prior(action, qs) ←───────────────────┘
```

---

### State Inference: `inference.py` + `algos.py`

**The inference pipeline updates posterior beliefs q(s) given observations.**

#### `inference.py` — High-Level API (15 functions)

| Function | Description |
|----------|-------------|
| `update_posterior_states` | Main entry point. Routes to FPI, MMP, or VMP based on `inference_algo` |
| `_run_one_step_inference` | Single-step FPI (fixed-point iteration) |
| `_run_sequence_inference` | Multi-step MMP (marginal message passing) |
| `joint_dist_factor` | Joint distribution over a single factor |
| `smoothing_ovf` | Overshoot variational filtering |
| `smoothing_exact` | Exact Bayesian smoothing |

#### `algos.py` — Algorithm Implementations (34 functions)

**Fixed-Point Iteration (FPI):**

| Function | Description |
|----------|-------------|
| `run_vanilla_fpi` | Standard FPI: iteratively refine q(s) until convergence |
| `run_factorized_fpi` | Factorized FPI for multi-factor models |
| `run_factorized_fpi_hybrid` | Hybrid FPI with padded arrays |
| `run_factorized_fpi_end2end_padded` | End-to-end padded FPI for JIT |

**Marginal Message Passing (MMP):**

| Function | Description |
|----------|-------------|
| `get_mmp_messages` | Compute MMP messages (forward + backward) |
| `run_mmp` | Full MMP inference with temporal smoothing |

**Variational Message Passing (VMP):**

| Function | Description |
|----------|-------------|
| `get_vmp_messages` | Compute VMP messages between factors |
| `run_vmp` | Full VMP inference |

**HMM Filtering & Smoothing (via associative scan):**

| Function | Description |
|----------|-------------|
| `hmm_filter_scan_rowstoch` | Row-stochastic HMM filtering with `jax.lax.associative_scan` |
| `hmm_smoother_scan_rowstoch` | Row-stochastic HMM smoothing |
| `hmm_filter_scan_colstoch` | Column-stochastic variant |
| `hmm_smoother_scan_colstoch` | Column-stochastic smoothing |
| `run_exact_single_factor_hmm_scan` | Exact single-factor HMM via scan |

**Variational Filtering:**

| Function | Description |
|----------|-------------|
| `variational_filtering_step` | Single step of variational filtering |
| `update_variational_filtering` | Full variational filtering loop |
| `run_online_filtering` | Online (streaming) inference |

---

### Policy Inference: `control.py`

**The control module computes the policy posterior q(π) by evaluating expected free energy (EFE) for each policy.**

#### Mathematical Foundation

```
q(π) = σ(−G(π))     where σ is softmax

G(π) = Σ_τ [ G_epistemic(π,τ) + G_pragmatic(π,τ) + G_novelty(π,τ) ]

G_epistemic(π,τ)  = −E_q(o|s,π)[ H[p(s|o)] − H[q(s|π)] ]    # Information gain
G_pragmatic(π,τ)  = −E_q(o|π)[ log p(o|C) ]                    # Expected utility
G_novelty(π,τ)    = −E_q(s|π)[ D_KL[q(A)||p(A)] ]              # Parameter info gain
```

#### Key Functions (18 total)

| Function | Description |
|----------|-------------|
| `construct_policies` | Enumerate all policy sequences of length `policy_len` |
| `update_posterior_policies` | Compute q(π) = σ(−G(π)) for all policies |
| `compute_neg_efe_policy` | Full −G(π) with epistemic + pragmatic + novelty |
| `compute_neg_efe_policy_inductive` | −G with inductive reachability bias |
| `update_posterior_policies_inductive` | q(π) with inductive I matrix |
| `compute_expected_state` | E_q[s\|π] = B(a) · q(s) |
| `compute_expected_obs` | E_q[o\|π] = A · E_q[s\|π] |
| `compute_info_gain` | Epistemic value: H[E_q[o]] − E_q[H[p(o\|s)]] |
| `compute_expected_utility` | Pragmatic value: E_q[o] · C |
| `calc_negative_pA_info_gain` | Parameter novelty for A learning |
| `calc_negative_pB_info_gain` | Parameter novelty for B learning |
| `generate_I_matrix` | Inductive inference reachability matrix |
| `sample_action` | Sample from action marginals |
| `sample_policy` | Sample entire policy sequences |
| `get_marginals` | Marginalize q(π) to per-factor action probs |

---

### Learning: `learning.py`

**Dirichlet parameter updates for A (observation likelihood) and B (transition dynamics).**

| Function | Description |
|----------|-------------|
| `update_obs_likelihood_dirichlet` | Update pA ← pA + lr · Δ(obs, qs) |
| `update_obs_likelihood_dirichlet_m` | Per-modality variant |
| `update_state_transition_dirichlet` | Update pB ← pB + lr · Δ(qs, qs', action) |
| `update_state_transition_dirichlet_f` | Per-factor variant |

**Learning rule:**
```
qA_m(o,s) = pA_m(o,s) + lr · obs_m(o) ⊗ q(s)     # Observation likelihood
qB_f(s',s,a) = pB_f(s',s,a) + lr · q(s') ⊗ q(s)   # Transition dynamics
```

---

### Mathematical Primitives: `maths.py`

**32 functions providing the mathematical backbone.**

#### Information-Theoretic

| Function | Description |
|----------|-------------|
| `stable_entropy(x)` | H(x) = −Σ x log x (numerically stable) |
| `stable_cross_entropy(x, y)` | H(x,y) = −Σ x log y |
| `stable_xlogx(x)` | x log x with 0·log(0) = 0 |
| `log_stable(x)` | log(x + ε) with machine epsilon |
| `dirichlet_kl_divergence(q, p)` | KL(Dir(q) ‖ Dir(p)) |

#### Tensor Operations

| Function | Description |
|----------|-------------|
| `factor_dot(A, qs)` | Multi-factor tensor contraction A · (q₁ ⊗ q₂ ⊗ ...) |
| `factor_dot_flex(ll, qs, dims)` | Flexible factor_dot with arbitrary dims |
| `spm_dot_sparse(sparse_A, qs)` | Sparse tensor contraction |
| `multidimensional_outer(arrs)` | Outer product of multiple arrays |

#### Variational Free Energy

| Function | Description |
|----------|-------------|
| `calc_vfe(qs, prior, ll, ...)` | Full VFE: F = E_q[log q(s)] − E_q[log p(o,s)] |
| `compute_accuracy(qs, A, obs)` | Accuracy: E_q[log p(o\|s)] |

#### SPM Compatibility

| Function | Description |
|----------|-------------|
| `spm_wnorm(A)` | SPM-style column normalization |
| `dirichlet_expected_value(dir_arr)` | E[Dir(α)] = α / Σα |

---

### Utilities: `utils.py`

**31 array construction and normalization helpers.**

| Category | Functions |
|----------|-----------|
| **Array builders** | `random_A_array`, `random_B_array`, `create_controllable_B`, `list_array_uniform`, `list_array_zeros`, `list_array_scaled` |
| **Normalization** | `norm_dist`, `list_array_norm_dist`, `validate_normalization` |
| **Dependency resolvers** | `resolve_a_dependencies`, `resolve_b_dependencies`, `resolve_b_action_dependencies` |
| **Indexing** | `get_combination_index`, `index_to_combination` |
| **Agent spec generation** | `generate_agent_spec`, `generate_agent_specs_from_parameter_sets` |
| **Categorical helpers** | `random_factorized_categorical`, `get_sample_obs` |
| **Block diagonal** | `build_block_diag_A`, `preprocess_A_for_block_diag`, `compute_log_likelihoods_block_diag` |

---

### Environments: `envs/`

#### Base Class: `PymdpEnv(Env)`

All environments inherit from `PymdpEnv` and implement:

| Method | Description |
|--------|-------------|
| `step(rng_key, action)` | Advance state, return new observations |
| `reset(rng_key)` | Reset to initial state |
| `A`, `B`, `D` | Generative model parameters (class attributes) |
| `A_dependencies`, `B_dependencies` | Dependency structure |

#### Environment Implementations

| Class | Module | States | Obs Modalities | Controls | Description |
|-------|--------|--------|----------------|----------|-------------|
| `TMaze` | `tmaze.py` | 2 factors (location×context) | 3 (location, reward, cue) | [5, 1] | T-maze with hidden reward location |
| `SimplifiedTMaze` | `tmaze.py` | Same | Same | Same | Parameter-simplified variant |
| `GeneralizedTMazeEnv` | `generalized_tmaze.py` | Multi-arm maze | Multi-modal | Per-axis | Maze parsed from ASCII matrix |
| `CueChainingEnv` | `cue_chaining.py` | Chain of cue locations | Per-cue modality | Location | Agent follows cue chain to reward |
| `GraphEnv` | `graph_worlds.py` | Graph nodes | Node identity | Neighbor edges | Cluster-connected graph navigation |
| `GridWorld` | `grid_world.py` | N×M grid cells | Cell identity | [up, down, left, right, stay] | 2D grid with optional walls |

#### Rollout Engine: `rollout.py`

```python
def rollout(agent, env, num_timesteps, rng_key, policy_search_fn=None) -> (env, info):
```

**The rollout function orchestrates the full Active Inference loop:**

```
for t in range(T):
    obs ← env.step(action)                    # Environment dynamics
    qs  ← agent.infer_states(obs, prior)       # Belief update
    q_pi, neg_efe ← agent.infer_policies(qs)   # Policy evaluation
    action ← agent.sample_action(q_pi, key)    # Action selection
    prior ← agent.update_empirical_prior(action, qs)  # Next-step prior

    # Optional online learning:
    pA ← update_obs_likelihood(pA, obs, qs)
    pB ← update_state_transition(pB, qs, action)
```

**Return `info` dict:**

| Key | Shape | Description |
|-----|-------|-------------|
| `qs` | `list[Array(batch, T, states)]` per factor | Posterior beliefs per timestep |
| `qpi` | `Array(batch, T, num_policies)` | Policy posterior per timestep |
| `neg_efe` | `Array(batch, T, num_policies)` | Negative EFE per policy per timestep |
| `action` | `Array(batch, T, num_factors)` | Selected actions per timestep |
| `obs` | `list[Array(batch, T, obs_dim)]` | Observations per timestep |
| `env_states` | varies | Environment state trajectory |

---

### Planning: `planning/`

#### Sophisticated Inference: `si.py`

**Tree-based policy search that evaluates multi-step consequences.**

```python
def si_policy_search(agent, qs, rng_key, max_nodes, max_branching,
                     pruning_threshold, entropy_threshold) -> dict
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_nodes` | 512 | Maximum tree size |
| `max_branching` | 32 | Actions per node |
| `pruning_threshold` | 0.01 | Min probability to keep branch |
| `entropy_threshold` | 0.1 | Entropy stopping criterion |

**Architecture:**
```
Tree(eqx.Module)
├── beliefs: list[Array]      # q(s) at each node
├── actions: Array             # Action taken to reach node
├── neg_efe: Array             # −G accumulated at node
├── parent: Array              # Parent node index
└── depth: Array               # Depth in tree
```

#### MCTS: `mcts.py`

**Monte Carlo Tree Search via the `mctx` library.**

```python
def mcts_policy_search(agent, qs, rng_key, num_simulations, depth) -> dict
```

Uses `mctx.gumbel_muzero_policy` with a custom `recurrent_fn` that implements Active Inference dynamics.

---

### Legacy Layer: `legacy/`

**NumPy-era backward compatibility layer. Mirrors the JAX API with NumPy arrays.**

| Module | Description |
|--------|-------------|
| `legacy/agent.py` | `Agent` class with `infer_states`, `infer_policies`, `sample_action` |
| `legacy/algos/` | FPI, MMP implementations in NumPy |
| `legacy/envs/` | `TMazeEnv`, `GridWorldEnv` in NumPy |
| `legacy/maths.py` | `spm_log_single`, `softmax`, etc. |
| `legacy/utils.py` | `sample()`, array builders |
| `legacy/control.py` | EFE computation, policy construction |
| `legacy/default_models.py` | Pre-built models (epistemic MAB, etc.) |

---

## Part II: Docxology Sidecar Architecture

### Module Map

```
docxology/
├── run_all.py                           # Pipeline entry (32 examples)
├── pkg/support/
│   ├── bootstrap.py                     # OrchestrationConfig, CLI, path setup
│   ├── mirror_dispatch.py               # 32 handlers + auto-viz + diagnostics
│   ├── patterns.py                      # Reusable pymdp call patterns
│   ├── viz.py                           # 18 visualization functions
│   ├── analysis.py                      # Entropy, KL, marginalization
│   ├── si_fixtures.py                   # SI/MCTS model builders
│   └── path_hack.py                     # sys.path management
├── tests/                               # 41 tests
├── docs/                                # This directory
├── manifests/                           # CI/nightly/legacy path lists
└── output/                              # Generated: 130+ files
```

### `bootstrap.py` — Configuration

#### `OrchestrationConfig` (frozen dataclass)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fast` | `bool` | `False` | Shorter loops (2–3 vs 15–25 timesteps) |
| `skip_heavy` | `bool` | `False` | Skip torch, long SI/MCTS, pybefit |
| `seed` | `int` | `0` | JAX PRNG seed |
| `verbose` | `bool` | `False` | Verbose logging |
| `output_dir` | `Path \| None` | `None` | Output directory for plots/data |

**Derived:** `save_plots → True` when `output_dir` is not None.

### `patterns.py` — Reusable Call Patterns

| Function | Purpose | Returns |
|----------|---------|---------|
| `random_agent_one_cycle(seed, policy_len, inference_algo)` | Build random model, run 1 cycle | `{qs_shapes, action, A_matrix, B_matrix, C_matrix, D_matrix, diagnostics}` |
| `complex_action_dependency_agent(seed)` | 3-factor agent with B_action_dependencies | `Agent` instance |

### `viz.py` — 18 Visualization Functions

#### Generative Model Plots
- `plot_likelihood_matrix(A)` — A matrix heatmap (viridis)
- `plot_transition_matrix(B)` — B matrix heatmap (viridis)
- `plot_empirical_prior(D)` — D bar chart (skyblue)
- `plot_prior_preferences(C)` — C bar chart (salmon)

#### Inference Plots
- `plot_beliefs_heatmap(qs_seq)` — q(s) states × timesteps (plasma)
- `plot_free_energy(F_seq)` — VFE line plot (tomato)
- `plot_action_probabilities(probs)` — Action bar chart (steelblue)
- `plot_policy_posterior(q_pi)` — q(π) bar chart (mediumpurple)
- `plot_efe_components(g_epi, g_prag)` — EFE breakdown (grouped bars)

#### Trajectory Analytics
- `plot_entropy_trajectory(qs_seq)` — H[q(s)] over time
- `plot_kl_divergence_trajectory(qs_seq, prior)` — KL(q‖D) over time
- `plot_efe_trajectory(neg_efe_seq)` — Best/mean −G with fill
- `plot_neg_efe_heatmap(neg_efe_seq)` — −G per policy × timestep (viridis)
- `plot_policy_posterior_heatmap(q_pi_seq)` — q(π) evolution (magma)
- `plot_action_frequency_donut(actions)` — Donut chart

#### Animation
- `plot_belief_trajectory_animation(qs_seq, dir, stem, fps=4)` — GIF

### `analysis.py` — Numerical Utilities

| Function | Description |
|----------|-------------|
| `compute_entropy(probs, axis)` | Shannon entropy H = −Σ p log p |
| `trajectory_divergence(qs_seq, prior)` | Per-timestep KL(q‖prior) |
| `marginalize_actions(q_pi, policies)` | Policy posterior → action marginals |
| `compute_accuracy_complexity(q_s, p_o_s, obs, prior)` | VFE = accuracy − complexity |

### `mirror_dispatch.py` — Handler Architecture

#### Dispatch Flow

```
run_all.py → mirror_dispatch.run_registered(script_file, cfg)
  → HANDLERS[key](cfg: OrchestrationConfig) → dict
    → _auto_plot_metrics(cfg, info, stem)
    → _extract_rollout_diagnostics(info)
    → returns {ok, id, diagnostics, ...}
```

#### `_auto_plot_metrics` Trigger Logic

| Data Key | Required Shape | Visualizations |
|----------|---------------|----------------|
| `A_matrix` | list of arrays | A heatmap, B heatmap, D bar, C bar |
| `qs` (list of factors `(B,T,N)`) | ndim ≥ 3 | Beliefs heatmap, entropy, KL, belief anim GIF |
| `qs` (list of per-step arrays) | ndim ≤ 2 | Same (legacy format) |
| `qpi` | `(B,T,N_π)` | Policy posterior heatmap |
| `neg_efe` | `(B,T,N_π)` | EFE trajectory, neg-EFE heatmap |
| `action` | `(B,T,N_f)` | Action frequency donut |
| `D_matrix` + `qs` | both present | KL from prior |

#### Handler Categories (32 total)

| Category | Handlers | Pattern |
|----------|---------|---------|
| Rollout (JAX) | tmaze, gen_tmaze, cue_chain, graph, knapsack, pybefit | `rollout()` → `_extract_rollout_diagnostics()` |
| Legacy (NumPy) | agent_demo, tmaze, tmaze_learning, grid1, grid2 | Manual per-step loop |
| Single-shot | model_construction, complex_action, neural, infer_states, learning, inductive×2, sparse, free_energy, docx×4, SI×3, MCTS×2 | One-cycle or specialized |
| Upstream delegate | chained_cue_nav, tmaze_recoverability | `runpy.run_path()` |

---

## Part III: Active Inference Mathematical Framework

### Generative Model

The agent maintains a **Partially Observable Markov Decision Process (POMDP)** as its generative model:

```
p(õ, s̃, π) = p(π) Π_τ [ p(o_τ|s_τ) p(s_τ|s_{τ-1}, a_τ) ]
            = p(π) Π_τ [ A · B(a_τ) ]
```

Where:
- **A** = p(o|s) — observation likelihood (how states generate observations)
- **B(a)** = p(s'|s,a) — transition dynamics (how actions change states)
- **C** = log p̃(o) — prior preferences (what observations the agent prefers)
- **D** = p(s₀) — initial state prior

### Perception: Variational Free Energy

**Goal:** Approximate posterior q(s) ≈ p(s|o) by minimizing VFE:

```
F[q] = E_q[log q(s)] − E_q[log p(o,s)]
     = D_KL[q(s) ‖ p(s)] − E_q[log p(o|s)]
     = Complexity − Accuracy
```

- **Accuracy**: How well beliefs explain observations
- **Complexity**: How far beliefs deviate from prior expectations

### Action: Expected Free Energy

**Goal:** Select policies π that minimize expected free energy G(π):

```
G(π) = Σ_τ E_q(o,s|π) [ log q(s|π) − log p(o,s) ]
     = Σ_τ [ −Epistemic(τ) − Pragmatic(τ) ]
```

**Epistemic value** (information gain):
```
−E_q(o|s,π)[ H[p(s|o)] − H[q(s|π)] ]  →  Drives exploration
```

**Pragmatic value** (expected utility):
```
−E_q(o|π)[ log p(o|C) ]  →  Drives goal-directed behavior
```

**Policy posterior:**
```
q(π) = σ(−G(π))  =  softmax over negative EFE
```

### Learning: Dirichlet Updates

**Goal:** Update model parameters to better predict observations:

```
pA(o,s) ← pA(o,s) + η · o ⊗ q(s)     # "I saw o when I believed s"
pB(s',s,a) ← pB(s',s,a) + η · q(s') ⊗ q(s)   # "s' followed s under a"
```

Where η is the learning rate. Expected parameters:
```
E[A] = pA / Σ_o pA     (normalized Dirichlet)
E[B] = pB / Σ_s' pB
```

### Inductive Inference: Reachability

The **I matrix** encodes reachable goal states:

```
I(s) = Σ_d P(reach goal | start at s, depth=d) > threshold
```

Used to bias policy selection toward states with high reachability:
```
G_inductive(π) = G(π) + γ · I · q(s|π)
```

### Sophisticated Inference

**Replaces flat policy enumeration with tree search:**

```
For each possible action sequence:
  1. Predict future observations: q(o|s,a) = A · B(a) · q(s)
  2. Update beliefs: q(s|o) via Bayesian update
  3. Evaluate −G at each tree node
  4. Prune low-probability branches
  5. Backpropagate values to root
```

---

## Part IV: Diagnostics Schema

### Modern Rollout Handlers (JAX)

```json
{
  "beliefs_per_timestep": [[0.03, 0.17, 0.80], ...],
  "actions_per_timestep": [2, 0, 1, ...],
  "observations_per_timestep": [[0], [2], ...],
  "q_pi_per_timestep": [[0.04, 0.96, ...], ...],
  "neg_efe_per_timestep": [[-1.2, 0.5, ...], ...]
}
```

### Legacy Handlers (NumPy)

```json
{
  "beliefs_per_timestep": [...],
  "actions_per_timestep": [...],
  "states_per_timestep": [[0, 0], [1, 0], ...],
  "q_pi_per_timestep": [...],
  "efe_per_timestep": [...],
  "vfe_per_timestep": [...]
}
```

### Single-Shot Handlers

```json
{
  "beliefs": [[0.03, 0.17, 0.80], [0.64, 0.36]],
  "q_pi": [0.32, 0.33, 0.35],
  "neg_efe": [0.88, 0.89, 0.97]
}
```

---

## Part V: Output Routing

| Output Type | Path Pattern | Description |
|-------------|-------------|-------------|
| JSON diagnostics | `output/{cat}/{stem}_data.json` | Per-timestep beliefs, VFE, EFE, q_pi, actions |
| Beliefs heatmap | `output/{cat}/{stem}_beliefs.png` | q(s) over time |
| Entropy trajectory | `output/{cat}/{stem}_entropy.png` | H[q(s)] over time |
| EFE trajectory | `output/{cat}/{stem}_efe_traj.png` | Best/mean −G |
| Neg-EFE heatmap | `output/{cat}/{stem}_efe_heatmap.png` | −G×policies×time |
| Policy posterior | `output/{cat}/{stem}_qpi_heatmap.png` | q(π) evolution |
| Belief animation | `output/{cat}/{stem}_beliefs_anim.gif` | Animated beliefs |
| Action donut | `output/{cat}/{stem}_action_donut.png` | Action frequency |
| KL divergence | `output/{cat}/{stem}_kl.png` | KL(q‖D) from prior |
| A/B/C/D matrices | `output/{cat}/{stem}_{A,B,C,D}.png` | Generative model |

---

## Part VI: Testing Matrix

| Test File | Tests | Focus |
|-----------|-------|-------|
| `test_orchestrations.py` | 32 | Every handler with `--fast --skip-heavy` |
| `test_viz.py` | 3 | Beliefs heatmap, free energy, action probs |
| `test_analysis.py` | 3 | Entropy, KL divergence, action marginalization |
| `test_manifests.py` | 2 | Manifest path existence, notebook listing |
| `test_notebook_runner_smoke.py` | 1 | Module help flag |
| **Total** | **41** | All passing |

---

## Part VII: Maintenance

### Adding a New Example

1. Create thin runner in `docxology/examples/{category}/{name}.py`
2. Add handler `_h_{name}(cfg) → dict` in `mirror_dispatch.py`
3. Register in `HANDLERS` dict
4. Add path to `manifests/orchestrations.txt`
5. Return `diagnostics` dict for JSON persistence
6. Call `_auto_plot_metrics(cfg, info, stem)` for auto-viz
7. Update `examples_catalog.md`
8. Update `validation_matrix.md`
9. Run `uv run pytest tests/`

### Path Conventions

- **Catalog/matrix:** relative to pymdp repo root
- **Handler keys:** relative to `docxology/examples/`
- **Manifest paths:** relative to `docxology/`

---

## See Also

- [`../AGENTS.md`](../AGENTS.md) — full docxology layout
- [`../../docs/AGENTS.md`](../../docs/AGENTS.md) — Sphinx build and extensions
