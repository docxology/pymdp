# Orchestrator Internals

Deep-dive technical reference for `pkg/support/mirror_dispatch.py` — the central execution engine that drives all 32 docxology examples.

---

## Module Role

`mirror_dispatch.py` is the **Thin Orchestrator** — it maps example script paths to handler functions, executes them with real pymdp APIs, and then runs a comprehensive post-processing pipeline that validates, serializes, visualizes, and reports on every execution.

The module contains **1185 lines** comprising:
- 32 handler functions (`_h_*`)
- 5 orchestrator support functions
- 1 handler registry (`HANDLERS` dict)
- 1 dispatch entry point (`run_registered`)

---

## Handler Registry

```python
HANDLERS: dict[str, Callable[[OrchestrationConfig], dict[str, Any]]] = {
    "examples/api/model_construction_tutorial.py":    _h_model_construction,
    "examples/advanced/complex_action_dependency.py": _h_complex_action_dependency,
    "examples/envs/tmaze_demo.py":                    _h_tmaze_rollout,
    # ... 29 more entries
}
```

Each handler:
1. Receives an `OrchestrationConfig` (seed, fast, skip_heavy, output_dir)
2. Constructs real pymdp objects (Agent, Env, etc.)
3. Executes real inference/planning/learning
4. Returns an `info` dict containing all tensors and diagnostics
5. Calls `_auto_plot_metrics(cfg, info, stem)` for automatic post-processing

---

## Dispatch Entry Point

```python
def run_registered(script_file: Path, cfg: OrchestrationConfig) -> dict[str, Any]:
```

1. Resolves `script_file` to a handler key via relative path from `docxology/`
2. Looks up the handler in `HANDLERS`
3. Calls the handler with `cfg`
4. Returns the handler's result dict

---

## The Post-Processing Pipeline

When `_auto_plot_metrics(cfg, info, stem)` is called, it triggers the following sequence:

### Step 1: Mathematical Invariant Validation

```python
def _verify_invariants(info: dict[str, Any]) -> dict[str, Any]:
```

Checks:
- `qs` (beliefs): final distribution sums to 1.0 over the last axis (±1e-3)
- `qpi` (policy posterior): final distribution sums to 1.0
- `A_matrix` (likelihood): columns sum to 1.0
- `B_matrix` (transitions): columns sum to 1.0

Returns `{"passed": bool, "violations": list[str]}`.

### Step 2: Retroactive Shannon Entropy Derivation

If `qs` is present but `H_qs` is not:

```python
for q in qs_raw:
    q_safe = np.clip(q_arr, 1e-12, 1.0)
    h = -np.sum(q_safe * np.log(q_safe))
    h_seq.append(float(h))
info["H_qs"] = h_seq
```

This ensures entropy trajectories are always available for Performance Insights, even when the original script doesn't compute them.

### Step 3: Unrestricted Tensor Serialization

```python
def _save_native_arrays(cfg: OrchestrationConfig, info: dict[str, Any], stem: str) -> None:
```

Iterates **all** keys in `info`:
- Skips keys starting with `_` (private/metadata)
- Skips dict values (nested structures)
- For list/tuple of arrays: saves each factor as `{key}_factor_{i}`
- For scalar arrays: saves directly
- Archives via `numpy.savez_compressed` → `{stem}_model_trace.npz`

### Step 4: Full JSON Serialization

```python
def _to_serializable(obj: Any) -> Any:
```

Recursively converts JAX arrays, NumPy arrays, and complex types to JSON-safe Python primitives. Writes `{stem}_full_data.json` with the complete info dict including derived metrics.

### Step 5: Conditional Visualization

Checks data keys and triggers appropriate viz functions:

| Check | Trigger | Function |
|---|---|---|
| `action` present, size > 0 | Action frequency bar chart | `plot_action_probabilities` |
| `action_probs` present | Action probability bar | `plot_action_probabilities` |
| `A_matrix` present | Likelihood heatmap | `plot_likelihood_matrix` |
| `B_matrix` present | Transition heatmap | `plot_transition_matrix` |
| `D_matrix` present | Prior bar chart | `plot_empirical_prior` |
| `C_matrix` present | Preferences bar chart | `plot_prior_preferences` |
| `q_pi` present | Policy posterior bar | `plot_policy_posterior` |
| `G_epistemic` + `G_pragmatic` | EFE breakdown | `plot_efe_components` |
| `qs` with ≥2 timesteps | Beliefs heatmap | `plot_beliefs_heatmap` |
| `qs` with ≥2 timesteps | Entropy trajectory | `plot_entropy_trajectory` |
| `qs` + `D_matrix` | KL from prior | `plot_kl_divergence_trajectory` |
| `qs` with ≥3 timesteps | Belief animation | `plot_belief_trajectory_animation` |
| `qpi` with T≥2, π≥2 | Policy posterior heatmap | `plot_policy_posterior_heatmap` |
| `neg_efe` with T≥2, π≥2 | EFE trajectory + heatmap | `plot_efe_trajectory`, `plot_neg_efe_heatmap` |
| `action` with ≥4 items, ≥2 unique | Donut chart | `plot_action_frequency_donut` |
| `vfe` or `F` with size > 1 | VFE line plot | `plot_free_energy` |
| `I_matrix` or `I` with ndim ≥ 2 | Reachability heatmap | `plot_reachability_matrix` |

### Step 6: Markdown Execution Report

```python
def _generate_markdown_report(cfg, stem, invariants, info) -> None:
```

Assembles `{stem}_execution_report.md` with:

1. **Configuration** — seed, fast mode, skip_heavy
2. **Mathematical Invariants** — ✅/❌ status with violation details
3. **Performance Insights** — scalar table searching for:
   - `H_qs` → Shannon Entropy H(q)
   - `vfe` / `F` → Variational Free Energy F
   - `neg_efe` / `G` → Negative Expected Free Energy −G
   - `KL` → KL Divergence D_KL(q||p)
4. **Native Trace Archive** — link to `.npz` file
5. **Visualizations** — embeds all generated PNG files with descriptive headers

---

## Handler Patterns

### Pattern A: Rollout (JAX)

Used by: TMaze, GeneralizedTMaze, CueChaining, GraphWorlds, Knapsack, Pybefit

```python
def _h_tmaze_rollout(cfg):
    env = TMaze(...)
    agent = Agent(env.A, env.B, C=C, D=env.D, ...)
    _, info = rollout(agent, env, num_timesteps, rng_key)
    info = _extract_rollout_diagnostics(info)
    _auto_plot_metrics(cfg, info, stem)
    return {"ok": True, "id": "envs/tmaze_demo", ...}
```

### Pattern B: Single-Shot

Used by: Model construction, complex action, learning, inductive, sparse

```python
def _h_model_construction(cfg):
    detail = random_agent_one_cycle(seed=cfg.seed, ...)
    _auto_plot_metrics(cfg, detail, stem)
    return {"ok": True, "id": "api/model_construction_tutorial", "detail": detail, ...}
```

### Pattern C: Upstream Delegate

Used by: Chained cue navigation, TMaze recoverability

```python
def _h_chained_cue_nav_upstream(cfg):
    runpy.run_path(str(upstream_script), ...)
    return {"ok": True, "id": "envs/chained_cue_navigation", ...}
```

### Pattern D: Legacy (NumPy)

Used by: Agent demo, TMaze, TMaze learning, GridWorld ×2, Free energy

```python
def _h_legacy_tmaze(cfg):
    agent = legacy_Agent(...)
    for t in range(T):
        qs = agent.infer_states(obs)
        q_pi, efe = agent.infer_policies()
        action = agent.sample_action()
        # ... accumulate per-step diagnostics
    _auto_plot_metrics(cfg, info, stem)
    return {"ok": True, "id": "legacy/tmaze_demo", ...}
```

---

## See Also

- [AGENTS.md](AGENTS.md) — full API reference including handler categories table
- [visualization_reference.md](visualization_reference.md) — complete viz function signatures
- [docxology_pymdp_overview.md](docxology_pymdp_overview.md) — high-level overview
