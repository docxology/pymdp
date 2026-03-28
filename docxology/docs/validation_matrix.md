# Validation Matrix

Maps pymdp **capability areas** to **upstream** `test/test_*.py` modules, **representative examples**, Upstream Documentation (Sphinx & MkDocs), **diagnostic output** types, and **key pymdp APIs** exercised.

All 32 examples produce JSON diagnostic logs into `docxology/output/{category}/{stem}/`. Every example automatically generates:

- `{stem}_validation.json` — handler return dict
- `{stem}_full_data.json` — complete info dict with derived Shannon entropy (`H_qs`) and mathematical invariant audits (`_invariants`)
- `{stem}_model_trace.npz` — compressed NumPy archive of all tensor parameters (unrestricted extraction)
- `{stem}_execution_report.md` — auto-generated Markdown with config, invariant status, Performance Insights scalar table, and embedded visualization PNGs

Rollout-based examples additionally generate up to 13 visualization types (beliefs heatmap, entropy, EFE trajectory, neg-EFE heatmap, policy posterior heatmap, belief animation GIF, VFE trajectory, reachability matrix, A/B/C/D matrices, action donut, action bars, EFE breakdown, KL from prior).

---

## JAX Agent and Control

| Area                 | Tests                                    | Examples                                | Upstream Docs                                                                                                                                  | Diagnostics                                     | Key APIs                                                  |
| -------------------- | ---------------------------------------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------- |
| `Agent` construction | `test_agent_jax.py`, `test_agent.py`     | `api/model_construction_tutorial.py`    | `docs/agent.rst`, [MkDocs API](../../docs-mkdocs/api/agent.md), [MkDocs NB](../../docs-mkdocs/tutorials/notebooks/using_the_agent_class.ipynb) | beliefs, q_pi, neg_efe                          | `Agent()`, `utils.random_A_array`, `utils.random_B_array` |
| Policy inference     | `test_control_jax.py`, `test_control.py` | `advanced/complex_action_dependency.py` | `docs/control.rst`, [MkDocs API](../../docs-mkdocs/api/control.md)                                                                             | per-factor beliefs, q_pi (12 policies), neg_efe | `Agent.infer_policies()`, `Agent.sample_action()`         |
| Multi-factor control | `test_agent_jax.py`                      | `advanced/complex_action_dependency.py` | `docs/agent.rst`, [MkDocs API](../../docs-mkdocs/api/agent.md)                                                                                 | per-factor beliefs, full neg_efe landscape      | `B_action_dependencies`, `sampling_mode="full"`           |
| Action sampling      | `test_control_jax.py`                    | All env demos (#5–9)                    | `docs/control.rst`, [MkDocs API](../../docs-mkdocs/api/control.md)                                                                             | actions_per_timestep                            | `Agent.sample_action(q_pi, rng_key)`                      |

### Verified Behaviors

- Multi-factor models with cross-factor `B_action_dependencies` produce valid 12-policy landscapes
- Both `sampling_mode="full"` and default marginal sampling work correctly
- `num_controls` configuration properly shapes the policy space
- Mathematical invariants (`_verify_invariants`) confirm `qs` and `qpi` normalization
- Shannon entropy `H_qs` retroactively derived and logged in Performance Insights

---

## State Inference

| Area               | Tests                                        | Examples                                             | Upstream Docs                                                          | Diagnostics             | Key APIs                                                     |
| ------------------ | -------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------- | ----------------------- | ------------------------------------------------------------ |
| FPI algorithm      | `test_inference_jax.py`                      | `advanced/infer_states_optimization/methods_test.py` | `docs/inference.rst`, [MkDocs API](../../docs-mkdocs/api/inference.md) | per-algo beliefs, q_pi  | `Agent(inference_algo="fpi")`                                |
| MMP algorithm      | `test_mmp.py`, `test_message_passing_jax.py` | same                                                 | `docs/algos/mmp.rst`, [MkDocs API](../../docs-mkdocs/api/algos.md)     | per-algo beliefs, q_pi  | `Agent(inference_algo="mmp")`                                |
| Optimized dispatch | `test_infer_states_optimized.py`             | same                                                 | `docs/inference.rst`, [MkDocs API](../../docs-mkdocs/api/inference.md) | FPI vs MMP comparison   | `Agent.infer_states(obs, empirical_prior, return_info=True)` |
| VFE computation    | `test_vfe_jax.py`                            | `legacy/free_energy_calculation.py`                  | `docs/notebooks/free_energy_calculation.ipynb`                         | spm_log values, entropy | `spm_log_single()`, `compute_accuracy_complexity()`          |

### Verified Behaviors

- FPI and MMP converge to compatible beliefs on the same model
- `return_info=True` provides convergence metadata
- VFE decomposes correctly into accuracy − complexity
- Invariant validation confirms all posterior distributions properly normalized
- `H_qs` entropy trajectory derived retroactively where not explicitly logged

---

## Learning

| Area               | Tests                                       | Examples                                | Upstream Docs                                                        | Diagnostics                                   | Key APIs                                     |
| ------------------ | ------------------------------------------- | --------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------- | -------------------------------------------- |
| Dirichlet updates  | `test_learning_jax.py`                      | `learning/learning_gridworld.py`        | `docs/learning.rst`, [MkDocs API](../../docs-mkdocs/api/learning.md) | qA_delta, prior/posterior shapes, beliefs, lr | `learning.update_obs_likelihood_dirichlet()` |
| Legacy pA learning | `test_learning.py`                          | `legacy/tmaze_learning_demo.py`         | —                                                                    | pA_delta, per-step beliefs                    | `Agent.update_A()` (legacy)                  |
| Neural encoder     | `test_agent_jax.py`, `test_learning_jax.py` | `advanced/pymdp_with_neural_encoder.py` | —                                                                    | beliefs, q_pi, neg_efe                        | torch+JAX co-existence verified              |

### Verified Behaviors

- `lr=0.5` produces measurable `qA_delta` from a single update
- pA Dirichlet counts accumulate correctly over 30 timesteps
- torch import does not conflict with JAX backend

---

## Environments and Rollouts

| Area               | Tests                      | Examples                                                                 | Upstream Docs                                                           | Output Types       | Key APIs                                      |
| ------------------ | -------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------- | ------------------ | --------------------------------------------- |
| TMaze              | `test_tmaze_envs.py`       | `envs/tmaze_demo.py` (#5), `envs/generalized_tmaze_demo.py` (#6)         | `docs/env.rst`, [MkDocs Env](../../docs-mkdocs/api/envs-env.md)         | 6 viz types        | `TMaze(categorical_obs=False)`, `rollout()`   |
| Cue chaining       | `test_cue_chaining_env.py` | `envs/cue_chaining_demo.py` (#7), `envs/chained_cue_navigation.py` (#10) | `docs/notebooks/cue_chaining_demo.ipynb`                                | 6 viz types + GIFs | `CueChainingEnv(num_cues=2)`, `rollout()`     |
| Graph worlds       | `test_env.py`              | `envs/graph_worlds_demo.py` (#8)                                         | `docs/env.rst`, [MkDocs Env](../../docs-mkdocs/api/envs-env.md)         | 6 viz types        | `GraphWorld(graph_type="grid")`, `rollout()`  |
| Knapsack surrogate | `test_env.py`              | `envs/knapsack_demo.py` (#9)                                             | —                                                                       | 6 viz types        | `GridWorld(3,3)`, `rollout()`                 |
| Rollout loop       | `test_rollout_function.py` | #5–10, #31                                                               | `docs/env.rst`, [MkDocs Rollout](../../docs-mkdocs/api/envs-rollout.md) | JSON diagnostics   | `rollout(agent, env, num_timesteps, rng_key)` |

### Verified Behaviors

- `rollout()` returns `info` dict with `qs`, `qpi`, `neg_efe`, `action` arrays
- All env `step()` methods return valid observations for `infer_states()`
- Batch dimension (dim 0) consistently set to 1
- 25+ timestep rollouts produce stable belief trajectories

### Output Per Rollout Example

Each rollout example (#5–9, #31) produces:

| File                               | Content                                                         |
| ---------------------------------- | --------------------------------------------------------------- |
| `{stem}_validation.json`           | Handler diagnostics (ok, id, elapsed)                           |
| `{stem}_full_data.json`            | Full diagnostics + derived H_qs entropy + invariant audits      |
| `{stem}_model_trace.npz`           | Native NumPy archive of all tensor keys (qs, qpi, neg_efe, etc.)|
| `{stem}_execution_report.md`       | Auto Markdown report with Performance Insights + embedded PNGs  |
| `{stem}_beliefs.png`               | q(s) heatmap (states × timesteps)                               |
| `{stem}_entropy.png`               | H[q(s)] over time                                               |
| `{stem}_efe_traj.png`              | Best/mean −G over time                                          |
| `{stem}_efe_heatmap.png`           | −G per policy × timestep                                        |
| `{stem}_qpi_heatmap.png`           | q(π) per policy × timestep                                      |
| `{stem}_beliefs_anim.gif`          | Animated belief bar chart                                       |
| `{stem}_kl.png`                    | KL(q‖D) from prior (if D_matrix present)                       |

---

## Sophisticated Inference and Planning

| Area             | Tests                                 | Examples                                                                   | Upstream Docs                                                           | Diagnostics                            | Key APIs                                            |
| ---------------- | ------------------------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------- | -------------------------------------- | --------------------------------------------------- |
| SI policy search | `test_sophisticated_inference_jax.py` | `si_tmaze_SIvalidation.py`, `si_generalized_tmaze.py`, `si_graph_world.py` | `docs/inference.rst`, [MkDocs SI](../../docs-mkdocs/api/planning-si.md) | `action_probs` (4-dim), `q_pi_shape`   | `si_policy_search(max_nodes=512, max_branching=32)` |
| MCTS planning    | `test_sophisticated_inference_jax.py` | `mcts_generalized_tmaze.py`, `mcts_graph_world.py`                         | [MkDocs MCTS](../../docs-mkdocs/api/planning-mcts.md)                   | `action_probs`, `action_weights_shape` | `mcts_policy_search(num_simulations=128, depth=4)`  |

### Verified Behaviors

- SI produces valid softmax distributions over 4 actions
- MCTS with 128 simulations converges to reasonable action weights
- Both methods use the same `build_single_cue_tmaze_like_model()` fixture

---

## Inductive Inference

| Area                    | Tests                             | Examples                        | Upstream Docs                                                        | Diagnostics                                               | Key APIs                                            |
| ----------------------- | --------------------------------- | ------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------- |
| Reachability (I matrix) | `test_inductive_inference_jax.py` | `inductive_inference/*.py` (×2) | `docs/inference.rst`, [MkDocs API](../../docs-mkdocs/api/control.md) | `I_matrix`, `H_vector`, `B_shape`, 0.5 threshold, depth=4 | `control.generate_I_matrix(H, B, threshold, depth)` |

### Verified Behaviors

- Chain transition `B` matrix (3 states) produces valid reachability landscape
- Threshold=0.5, depth=4 configuration generates non-trivial I matrix
- `I_matrix` natively archived in `_model_trace.npz` for offline analysis
- 2D+ `I_matrix` automatically triggers `plot_reachability_matrix` heatmap visualization

---

## Sparse Backends

| Area            | Tests                        | Examples                     | Upstream Docs        | Diagnostics              | Key APIs                                                 |
| --------------- | ---------------------------- | ---------------------------- | -------------------- | ------------------------ | -------------------------------------------------------- |
| JAX BCOO sparse | `test_jax_sparse_backend.py` | `sparse/sparse_benchmark.py` | `docs/inference.rst` | timing, nnz=4, trace=8.0 | `jax.experimental.sparse.BCOO.fromdense()`, `.todense()` |

---

## Model Fitting (pybefit)

| Area               | Tests                           | Examples                                | Upstream Docs       | Diagnostics                   | Key APIs                        |
| ------------------ | ------------------------------- | --------------------------------------- | ------------------- | ----------------------------- | ------------------------------- |
| pybefit + rollout  | `test_pybefit_model_fitting.py` | `model_fitting/fitting_with_pybefit.py` | `docs/learning.rst` | Full TMaze rollout (10 steps) | `import pybefit`, `rollout()`   |
| SVI recoverability | `test_tmaze_recoverability.py`  | `model_fitting/tmaze_recoverability.py` | —                   | SVI metrics + plot            | `--svi-steps 40 --num-agents 2` |

### Verified Behaviors

- pybefit import does not conflict with pymdp JAX backend
- Real TMaze rollout succeeds with pybefit in environment
- SVI converges in 40 steps with 2 agents

---

## Utilities and Distributions

| Area                     | Tests                                                      | Examples                    | Upstream Docs                                                                           | Diagnostics         | Key APIs                                           |
| ------------------------ | ---------------------------------------------------------- | --------------------------- | --------------------------------------------------------------------------------------- | ------------------- | -------------------------------------------------- |
| `utils` (random tensors) | `test_utils_jax.py`, `test_utils.py`                       | Model construction tutorial | `docs/notebooks/pymdp_fundamentals.ipynb`, [MkDocs API](../../docs-mkdocs/api/utils.md) | shapes, jax_version | `utils.random_A_array()`, `utils.random_B_array()` |
| Categorical observations | `test_distribution.py`, `test_categorical_observations.py` | Most examples               | `docs/inference.rst`                                                                    | —                   | `categorical_obs=False` (one-hot)                  |
| HMM / associative scan   | `test_hmm_associative_scan.py`                             | Inference notebooks         | `docs/algos/index.rst`, [MkDocs API](../../docs-mkdocs/api/algos.md)                    | —                   | Internal MMP implementation                        |

---

## Legacy (NumPy-era)

| Area              | Tests                                        | Examples                                                     | Upstream Docs                                  | Diagnostics                             | Key APIs                                  |
| ----------------- | -------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------- | --------------------------------------- | ----------------------------------------- |
| Legacy agent      | `test_demos.py`                              | `legacy/agent_demo.py` (#23)                                 | `docs-mkdocs/legacy/index.md`                  | per-step qs, q_pi, EFE, actions, states | `pymdp.legacy.agent.Agent`                |
| Legacy T-maze     | `test_demos.py`, `test_tmaze_envs.py`        | `legacy/tmaze_demo.py` (#24), `tmaze_learning_demo.py` (#25) | —                                              | per-step qs, q_pi, EFE, VFE, actions    | `TMazeEnv`, `Agent.infer_states/policies` |
| Legacy grid world | `test_demos.py`, `test_grid_world_parity.py` | `legacy/gridworld_tutorial_{1,2}.py` (#26–27)                | —                                              | per-step qs, q_pi, EFE, VFE, actions    | `GridWorldEnv`                            |
| Free energy maths | `test_vfe_jax.py`, `test_utils.py`           | `legacy/free_energy_calculation.py` (#28)                    | `docs/notebooks/free_energy_calculation.ipynb` | input dist, spm_log, entropy            | `spm_log_single()`                        |

### Verified Behaviors

- All 6 legacy handlers capture per-timestep VFE when `agent.F` is available
- EFE (G) values are captured from `agent.infer_policies()` return
- State transitions use `utils.sample(B[f][:, s, action])` correctly
- Legacy handlers that log `VFE` or `F` arrays automatically trigger `plot_free_energy`
- Terminal VFE/EFE values reported in Performance Insights tables of execution reports
- All tensor parameters natively archived in `_model_trace.npz`

---

## Docxology Examples

| Area                   | Tests                    | Examples                                     | Diagnostics            | Delegates To               |
| ---------------------- | ------------------------ | -------------------------------------------- | ---------------------- | -------------------------- |
| Minimal one-cycle      | `test_orchestrations.py` | `docxology/01_minimal_discrete_timestep.py`  | beliefs, q_pi, neg_efe | `random_agent_one_cycle()` |
| Learning single-update | `test_orchestrations.py` | `docxology/02_learning_pA_single_update.py`  | qA_delta, beliefs, lr  | `_h_learning_gridworld()`  |
| Rollout cue-chaining   | `test_orchestrations.py` | `docxology/03_rollout_cue_chaining_short.py` | full rollout + 6 viz   | `_h_cue_chaining()`        |
| SI vanilla planning    | `test_orchestrations.py` | `docxology/04_si_cue_agent_vanilla_plan.py`  | q_pi, neg_efe, action  | `si_cue_agent()`           |

---

## SPM / Parity

| Area                 | Tests                                                 | Examples              | Notes                                           |
| -------------------- | ----------------------------------------------------- | --------------------- | ----------------------------------------------- |
| SPM numerical parity | `test_SPM_validation.py`, `test_grid_world_parity.py` | Legacy grid tutorials | Verifies JAX outputs match MATLAB SPM reference |

---

## Full Upstream Sweep

```bash
# From repository root
uv sync --group test
uv run pytest test -n 2 -m "not nightly"

# Or via helper
bash docxology/scripts/run_upstream_test_suite.sh
```

## Docxology Test Suite

```bash
# From docxology/
cd docxology && uv run pytest tests/ -v

# Result: 41 passed (32 handlers + 3 viz + 3 analysis + 2 manifests + 1 notebook)
```

---

## Orchestrator Pipeline Functions

The following functions in `pkg/support/mirror_dispatch.py` execute automatically for every example:

| Function | Purpose |
| --- | --- |
| `_verify_invariants(info)` | Audits normalization of `qs`, `qpi`, `A_matrix`, `B_matrix` (sum-to-1 within ±1e-3) |
| `H_qs` retroactive derivation | Computes Shannon entropy $H(q) = -\sum q \log q$ from belief sequences when not explicitly logged |
| `_save_native_arrays(cfg, info, stem)` | Unrestricted sweep: iterates all non-dict, non-private keys and archives via `numpy.savez_compressed` |
| `_to_serializable(obj)` | Recursive JAX/NumPy → JSON converter for `_full_data.json` |
| `_auto_plot_metrics(cfg, info, stem)` | Conditional trigger of all 13+ visualization types based on data key presence |
| `_generate_markdown_report(cfg, stem, invariants, info)` | Assembles `_execution_report.md` with config, invariants, Performance Insights scalar table, and embedded PNGs |

### Performance Insights Metrics

| Metric Key | Display Name | Source |
| --- | --- | --- |
| `H_qs` | Shannon Entropy $H(q)$ | Retroactively derived from `qs` |
| `vfe` / `F` | Variational Free Energy $F$ | Logged by legacy handlers |
| `neg_efe` / `G` | Negative Expected Free Energy $-G$ | From `infer_policies()` |
| `KL` | KL Divergence $D_{KL}(q\|\|p)$ | From trajectory analysis |
