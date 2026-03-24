# Examples Catalog

**32 validated examples** across 12 categories, all executed via `run_all.py` through `pkg.support.mirror_dispatch`. Every example produces a `_data.json` with diagnostics and metrics.

## Quick Run

```bash
cd docxology && uv run python run_all.py
```

**Deps:** `base` = editable `inferactively-pymdp` (JAX backend). `modelfit` = `pybefit`. `torch` = PyTorch. All examples use **real** JAX-based pymdp APIs — zero mocks.

---

## Category: API (1 example)

### #1 — `api/model_construction_tutorial.py`

**What it does:** Builds a random multi-factor generative model (`A`, `B`, `C`, `D` matrices), runs one `infer_states → infer_policies → sample_action` cycle via `random_agent_one_cycle`.

| Property | Value |
|----------|-------|
| Handler | `_h_model_construction` |
| Timesteps | 1 cycle |
| Deps | base |
| Diagnostics | `beliefs`, `q_pi`, `neg_efe` (nested in `detail`) |
| Visualizations | Action probability bar chart, A/B/C/D matrix heatmaps |
| Upstream tests | `test_agent_jax.py`, `test_utils_jax.py` |

---

## Category: Advanced (3 examples)

### #2 — `advanced/complex_action_dependency.py`

**What it does:** Constructs a 3-factor agent with `B_action_dependencies` (factors 0,1,2 with cross-factor control dependencies). Runs `infer_states` + `infer_policies` to verify multi-factor policy inference.

| Property | Value |
|----------|-------|
| Handler | `_h_complex_action_dependency` |
| Timesteps | 1 step |
| Diagnostics | Per-factor `beliefs`, full `q_pi` over 12 policies, `neg_efe` landscape |
| Visualizations | Beliefs heatmap (factor 0) |
| Upstream tests | `test_control_jax.py`, `test_agent_jax.py` |

### #3 — `advanced/infer_states_optimization/methods_test.py`

**What it does:** Runs the same random agent under both **FPI** (fixed-point iteration) and **MMP** (marginal message passing) inference algorithms, comparing their beliefs and action selections.

| Property | Value |
|----------|-------|
| Handler | `_h_infer_states_methods` |
| Algorithms | FPI, MMP |
| Diagnostics | Per-algorithm `beliefs`, `q_pi`, `neg_efe` |
| Visualizations | Per-algorithm action bar charts |
| Upstream tests | `test_infer_states_optimized.py`, `test_inference_jax.py` |

### #4 — `advanced/pymdp_with_neural_encoder.py`

**What it does:** Verifies PyTorch presence, then runs a standard `random_agent_one_cycle` to confirm torch+JAX co-existence.

| Property | Value |
|----------|-------|
| Handler | `_h_neural_encoder` |
| Deps | **torch** |
| Diagnostics | `beliefs`, `q_pi`, `neg_efe` |
| Upstream tests | `test_agent_jax.py`, `test_learning_jax.py` |

---

## Category: Environments (6 examples)

### #5 — `envs/tmaze_demo.py`

**What it does:** Full 25-step rollout on `TMaze` environment. Agent infers hidden reward location from cue observations, navigates to collect rewards.

| Property | Value |
|----------|-------|
| Handler | `_h_tmaze_rollout` |
| Env | `pymdp.envs.TMaze(categorical_obs=False)` |
| Timesteps | 25 |
| Controls | `[5, 1]` (location × reward) |
| Diagnostics | Full rollout: `qs`, `qpi`, `neg_efe`, `actions`, `observations` per timestep |
| Visualizations | Beliefs heatmap, entropy, EFE trajectory, neg-EFE heatmap, q_pi heatmap, belief animation GIF |
| Upstream tests | `test_tmaze_envs.py`, `test_agent_jax.py` |

### #6 — `envs/generalized_tmaze_demo.py`

**What it does:** Full 25-step rollout on `GeneralizedTMaze` (multiple reward arms).

| Property | Value |
|----------|-------|
| Handler | `_h_generalized_tmaze` |
| Timesteps | 25 |
| Diagnostics | Full rollout |
| Visualizations | All 6 viz types |
| Upstream tests | `test_tmaze_envs.py` |

### #7 — `envs/cue_chaining_demo.py`

**What it does:** 20-step rollout on `CueChainingEnv` with 2 cue levels. Tests epistemic chaining — agent must follow a chain of cues to find the reward.

| Property | Value |
|----------|-------|
| Handler | `_h_cue_chaining` |
| Timesteps | 20 |
| Diagnostics | Full rollout |
| Visualizations | All 6 viz types |
| Upstream tests | `test_cue_chaining_env.py`, `test_rollout_function.py` |

### #8 — `envs/graph_worlds_demo.py`

**What it does:** 25-step rollout on `GraphWorld` environment (graph-structured state space).

| Property | Value |
|----------|-------|
| Handler | `_h_graph_worlds` |
| Timesteps | 25 |
| Diagnostics | Full rollout |
| Visualizations | All 6 viz types |
| Upstream tests | `test_env.py`, `test_agent_jax.py` |

### #9 — `envs/knapsack_demo.py`

**What it does:** 25-step rollout using `GridWorld(3×3)` as a surrogate for the knapsack combinatorial task.

| Property | Value |
|----------|-------|
| Handler | `_h_knapsack_surrogate` |
| Timesteps | 25 |
| Diagnostics | Full rollout |
| Visualizations | All 6 viz types |
| Upstream tests | `test_control_jax.py`, `test_agent_jax.py` |

### #10 — `envs/chained_cue_navigation.py`

**What it does:** Delegates to upstream GIF generation script. Produces navigation animation GIFs showing agent movement through the cue-chaining environment.

| Property | Value |
|----------|-------|
| Handler | `_h_chained_cue_nav_upstream` |
| Timesteps | 15 |
| Deps | base + Pillow |
| Output | `chained_cue_navigation_v1.gif`, `_v2.gif` |
| Upstream tests | `test_cue_chaining_env.py`, `test_rollout_function.py` |

---

## Category: Sophisticated Inference (5 examples)

### #11–13 — SI Policy Search (×3)

**What they do:** Run `si_policy_search` on a single-cue T-maze model with configurable horizon, max nodes, branching factor, pruning thresholds, and entropy stopping.

| # | Handler Key | Max Nodes | Max Branching |
|---|------------|-----------|---------------|
| 11 | `si_tmaze_SIvalidation.py` | 512 | 32 |
| 12 | `si_generalized_tmaze.py` | 512 | 32 |
| 13 | `si_graph_world.py` | 512 | 32 |

**Diagnostics:** `action_probs` (4-dim softmax), `q_pi_shape`. **Upstream:** `test_sophisticated_inference_jax.py`.

### #14–15 — MCTS Planning (×2)

**What they do:** Run `mcts_policy_search` with 128 simulations, depth 4. Return action weights.

| # | Handler Key | Simulations | Depth |
|---|------------|-------------|-------|
| 14 | `mcts_generalized_tmaze.py` | 128 | 4 |
| 15 | `mcts_graph_world.py` | 128 | 4 |

**Diagnostics:** `action_probs`, `action_weights_shape`. **Upstream:** `test_sophisticated_inference_jax.py`.

---

## Category: Inductive Inference (2 examples)

### #16–17 — Reachability Matrices

**What they do:** Compute the `I` (reachability) matrix using `pymdp.control.generate_I_matrix` with a chain transition model. Parameters: threshold=0.5, depth=4.

**Diagnostics:** `I_matrix` (full values), `H_vector`, `B_shape`, `threshold`, `depth`. **Upstream:** `test_inductive_inference_jax.py`.

---

## Category: Inference & Learning (2 examples)

### #18 — `inference_and_learning/inference_methods_comparison.py`

**What it does:** Wraps `_h_infer_states_methods` comparing FPI vs MMP on the same random model.

**Diagnostics:** Per-algo dict with `beliefs`, `q_pi`, `neg_efe`. **Upstream:** `test_inference_jax.py`, `test_message_passing_jax.py`.

### #19 — `learning/learning_gridworld.py`

**What it does:** Runs Dirichlet parameter update via `learning.update_obs_likelihood_dirichlet`. Computes posterior qA from prior pA given observations and beliefs.

| Property | Value |
|----------|-------|
| Learning rate | 0.5 |
| State space | 3 states, 4 observations |
| Diagnostics | `qA_delta`, `pA_prior_shape`, `qA_posterior_shape`, `beliefs`, `lr` |
| Upstream tests | `test_learning_jax.py`, `test_agent_jax.py` |

---

## Category: Model Fitting (2 examples)

### #20 — `model_fitting/fitting_with_pybefit.py`

**What it does:** With `pybefit` loaded, runs a **real TMaze rollout** (10 timesteps) to verify the full pybefit+pymdp stack works together.

| Property | Value |
|----------|-------|
| Deps | **modelfit** (pybefit) |
| Timesteps | 10 |
| Diagnostics | Full rollout via `_extract_rollout_diagnostics` |
| Visualizations | Beliefs, entropy, EFE trajectory, q_pi heatmap |
| Upstream tests | `test_pybefit_model_fitting.py` |

### #21 — `model_fitting/tmaze_recoverability.py`

**What it does:** Delegates to upstream SVI recoverability script with `--num-blocks 2 --svi-steps 40 --num-agents 2`.

| Property | Value |
|----------|-------|
| Deps | **modelfit** |
| SVI Steps | 40 |
| Output | `tmaze_recoverability_plot.png`, `_metrics.json` |
| Upstream tests | `test_tmaze_recoverability.py` |

---

## Category: Sparse (1 example)

### #22 — `sparse/sparse_benchmark.py`

**What it does:** Creates a 4×4 identity as JAX `BCOO` sparse, performs sparse addition, converts to dense.

**Diagnostics:** `sparse_ops_elapsed_s`, `nnz_input`, `dense_shape`, `result_trace`. **Upstream:** `test_jax_sparse_backend.py`.

---

## Category: Legacy (6 examples)

All legacy handlers use `pymdp.legacy.agent.Agent` (NumPy-era API).

### #23 — `legacy/agent_demo.py`

Multi-arm bandit (MAB) with epistemic model. 15 timesteps. **Diagnostics:** per-step `actions`, `states`, `beliefs`, `q_pi`, `EFE`.

### #24 — `legacy/tmaze_demo.py`

T-maze with reward preferences `C[1][1]=3, C[1][2]=-3`. 25 timesteps. **Diagnostics:** per-step `qs`, `q_pi`, `EFE`, `VFE`, `actions`.

### #25 — `legacy/tmaze_learning_demo.py`

T-maze with `pA` Dirichlet learning (lr=2.0). 30 timesteps. **Diagnostics:** per-step `qs`, `q_pi`, `EFE`, `VFE`, `actions` + final `pA_delta`.

### #26–27 — `legacy/gridworld_tutorial_{1,2}.py`

Grid world (3×3) with controllable transitions. 15 timesteps each. **Diagnostics:** per-step `qs`, `q_pi`, `EFE`, `VFE`, `actions`.

### #28 — `legacy/free_energy_calculation.py`

Computes `spm_log_single` on `[0.1, 0.9]`. **Diagnostics:** `input_distribution`, `spm_log_values`, `log_sum`, `entropy`.

**Upstream for all legacy:** `test_demos.py`, `test_grid_world_parity.py`, `test_vfe_jax.py`.

---

## Category: Docxology (4 examples)

### #29 — `docxology/01_minimal_discrete_timestep.py`

Delegates to `random_agent_one_cycle`. **Diagnostics:** `beliefs`, `q_pi`, `neg_efe` (in `detail`).

### #30 — `docxology/02_learning_pA_single_update.py`

Delegates to `_h_learning_gridworld`. **Diagnostics:** `qA_delta`, `beliefs`, `lr`.

### #31 — `docxology/03_rollout_cue_chaining_short.py`

Delegates to `_h_cue_chaining`. Full 20-step rollout with all 6 viz types. **Diagnostics:** full rollout.

### #32 — `docxology/04_si_cue_agent_vanilla_plan.py`

Builds SI cue agent, runs `infer_policies`, samples action. **Diagnostics:** `q_pi` (4-dim), `neg_efe` (4-dim), `sampled_action`.

---

## Upstream Test Coverage

| Test Module | Examples Covered | Focus |
|------------|-----------------|-------|
| `test_agent_jax.py` | #1, #2, #3, #4, #5–9, #19, #20 | Agent construction, policies |
| `test_control_jax.py` | #2, #9 | Policy inference, control |
| `test_inference_jax.py` | #3, #18 | Variational inference algos |
| `test_message_passing_jax.py` | #18 | MMP algorithm |
| `test_rollout_function.py` | #5–10, #31 | Batched rollout loop |
| `test_tmaze_envs.py` | #5, #6, #24 | T-maze env API |
| `test_cue_chaining_env.py` | #7, #10, #31 | Cue chaining env |
| `test_sophisticated_inference_jax.py` | #11–15 | SI + MCTS planning |
| `test_inductive_inference_jax.py` | #16, #17 | Reachability matrices |
| `test_learning_jax.py` | #19, #30 | Dirichlet learning |
| `test_pybefit_model_fitting.py` | #20 | pybefit integration |
| `test_tmaze_recoverability.py` | #21 | SVI parameter recovery |
| `test_jax_sparse_backend.py` | #22 | Sparse ops |
| `test_demos.py` | #23–28 | Legacy agent demos |
| `test_vfe_jax.py` | #28 | VFE computation |

---

## Visualization Types Generated

| Viz Type | File Suffix | Count | Generated By |
|----------|-------------|-------|-------------|
| Beliefs heatmap | `_beliefs.png` | 13 | `plot_beliefs_heatmap` |
| Entropy trajectory | `_entropy.png` | 6 | `plot_entropy_trajectory` |
| EFE trajectory | `_efe_traj.png` | 6 | `plot_efe_trajectory` |
| Neg-EFE heatmap | `_efe_heatmap.png` | 6 | `plot_neg_efe_heatmap` |
| Policy posterior | `_qpi_heatmap.png` | 6 | `plot_policy_posterior_heatmap` |
| Belief animation | `_beliefs_anim.gif` | 6 | `plot_belief_trajectory_animation` |
| KL from prior | `_kl.png` | 1 | `plot_kl_divergence_trajectory` |
| A/B/C/D matrices | `_A.png` etc. | varies | `plot_likelihood_matrix` etc. |
| Action bars | `_action.png` | varies | `plot_action_probabilities` |

---

## Running Tiers

```bash
# Full 32-example pipeline (from docxology/)
uv run python run_all.py

# CI-tier notebooks
uv run python docxology/scripts/run_docxology_notebooks.py

# Nightly set
uv run python docxology/scripts/run_docxology_notebooks.py docxology/manifests/notebooks_nightly.txt

# Upstream test suite
bash docxology/scripts/run_upstream_test_suite.sh
```

---

## Sphinx-only Notebooks

| Sphinx Path | Topic |
|-------------|--------|
| `docs/notebooks/pymdp_fundamentals.ipynb` | Package fundamentals |
| `docs/notebooks/active_inference_from_scratch.ipynb` | Active inference walkthrough |
| `docs/notebooks/using_the_agent_class.ipynb` | `Agent` API usage |
