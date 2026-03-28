# Visualization Reference

Complete reference for all 21 visualization functions in `pkg/support/viz.py`. All functions are headless-safe (`MPLBACKEND=Agg`) and produce publication-ready plots at 150 DPI.

---

## Global Aesthetics

Applied via `_apply_aesthetics()` at module load:

| Setting | Value |
|---|---|
| Style | `seaborn-v0_8-darkgrid` (graceful fallback) |
| Font size | 12pt (body), 14pt (axes titles), 16pt (figure title) |
| Figure background | `#f8f9fa` |
| Axes background | `#ffffff` |
| Grid | `#cccccc` at 0.6 alpha |
| Axes edge | `#333333` at 1.2pt width |

## Save Function

```python
def save_current_figure(path: Path | None, *, stem: str, suffix: str = ".png") -> Path | None:
```

Creates output directory if needed, saves at 150 DPI with tight bounding box, closes figure. Returns output path or `None`.

---

## Generative Model Plots

### `plot_likelihood_matrix(A_matrix, title)`

**Purpose:** Visualize the observation likelihood matrix A = P(o|s).

- Extracts modality 0 from list, squeezes to first control slice if 3D+
- Colormap: `viridis`
- Axes: X = Hidden States, Y = Observations

### `plot_transition_matrix(B_matrix, title)`

**Purpose:** Visualize the state transition matrix B = P(s'|s, a) for control 0.

- Extracts factor 0 from list, takes `[:, :, 0]` if 3D
- Colormap: `viridis`
- Axes: X = Current State, Y = Next State

### `plot_empirical_prior(D_matrix, title)`

**Purpose:** Bar chart of the initial state prior D = P(s₀).

- Extracts factor 0
- Color: `skyblue` with `steelblue` edge
- Axes: X = State Index, Y = Probability

### `plot_prior_preferences(C_matrix, title)`

**Purpose:** Bar chart of log-preferences C = log P̃(o).

- Extracts modality 0
- Colors: `salmon` (positive), `lightcoral` (negative) with `crimson` edge
- Axes: X = Observation Index, Y = Log Preference

---

## Inference Plots

### `plot_beliefs_heatmap(qs_seq, title)`

**Purpose:** Heatmap of posterior beliefs q(s) across timesteps.

- Input: list of 1D arrays (one per timestep)
- Stacks into matrix: states × timesteps
- Colormap: `plasma`, range [0, 1]
- Axes: X = Timestep, Y = State Index

### `plot_free_energy(F_seq, title)`

**Purpose:** Line plot of Variational Free Energy F over time.

- Input: list/array of scalar VFE values
- Color: `tomato` with `salmon` fill to zero
- Axes: X = Timestep, Y = Free Energy (nats)

### `plot_action_probabilities(action_probs, title)`

**Purpose:** Bar chart of action probabilities or frequencies.

- Squeezes input to 1D
- Color: `steelblue` with `navy` edge
- Axes: X = Action Index, Y = Probability

### `plot_policy_posterior(q_pi, title)`

**Purpose:** Bar chart of the policy posterior q(π).

- Color: `mediumpurple` with `rebeccapurple` edge
- Axes: X = Policy Index, Y = Probability

### `plot_efe_components(g_epistemic, g_pragmatic, title)`

**Purpose:** Grouped bar chart breaking down EFE into epistemic and pragmatic components.

- Colors: `cornflowerblue` (epistemic), `coral` (pragmatic)
- Axes: X = Policy Index, Y = −G Component Value

---

## Trajectory Analytics

### `plot_entropy_trajectory(qs_seq, title)`

**Purpose:** Line plot of Shannon entropy H[q(s)] over time.

- Computes: `H = −Σ q·log(q)` per timestep
- Color: `darkorange` with `moccasin` fill
- Axes: X = Timestep, Y = Entropy (nats)

### `plot_kl_divergence_trajectory(qs_seq, prior, title)`

**Purpose:** Line plot of KL divergence KL(q||D) from the initial prior over time.

- Computes: `KL = Σ q·(log q − log p)` per timestep
- Color: `crimson` with `mistyrose` fill
- Axes: X = Timestep, Y = KL Divergence (nats)

### `plot_efe_trajectory(neg_efe_seq, title)`

**Purpose:** Line plot of negative Expected Free Energy −G per policy over time.

- Input: list of lists `[[−G_π₁, −G_π₂, ...], ...]`
- Plots: best (max) and mean −G across policies
- Colors: `steelblue` (mean), `navy` (best)
- Fill between with `lightskyblue`
- Axes: X = Timestep, Y = −G

### `plot_neg_efe_heatmap(neg_efe_seq, title)`

**Purpose:** Heatmap of −G values per policy per timestep.

- Input: list of lists
- Colormap: `viridis`
- Axes: X = Timestep, Y = Policy Index

### `plot_policy_posterior_heatmap(q_pi_seq, title)`

**Purpose:** Heatmap of q(π) evolution over time.

- Input: list of lists
- Colormap: `magma`, range [0, 1]
- Axes: X = Timestep, Y = Policy Index

### `plot_action_frequency_donut(actions, title)`

**Purpose:** Donut chart of action frequency distribution.

- Computes unique action counts from flat array
- Uses colormap: `Set2`
- Center: white circle (donut hole)
- Labels: `Action {i}: {count}`

---

## Reachability and Specialized

### `plot_reachability_matrix(I_matrix, title)`

**Purpose:** Heatmap of the inductive reachability matrix I.

- If list/tuple, extracts first element
- Squeezes and takes 2D slice if needed
- Colormap: `YlOrRd`
- Axes: X = State Index, Y = State/Depth Index

### `plot_spm_log_comparison(x, log_x, title)`

**Purpose:** Side-by-side comparison of standard log vs SPM log.

- Subplot 1: `log(x)` (standard) in `steelblue`
- Subplot 2: `spm_log(x)` in `coral`
- Axes: X = Index, Y = Log Value

### `plot_sparse_structure(dense_matrix, nnz, title)`

**Purpose:** Binary heatmap showing sparse matrix structure.

- Converts to binary (nonzero = 1)
- Colormap: `Blues`
- Title appended with `(nnz={nnz})`
- Axes: X = Column, Y = Row

---

## Animation

### `plot_belief_trajectory_animation(qs_seq, output_dir, stem, fps=4)`

**Purpose:** Animated GIF showing belief bar chart evolving over time.

- One frame per timestep
- Color: `steelblue` bars
- Saves: `{output_dir}/{stem}_beliefs_anim.gif`
- Uses `Pillow` writer at `fps` frames per second

---

## Auto-Trigger Summary

The orchestrator in `mirror_dispatch.py` automatically triggers these functions based on data presence:

```text
_auto_plot_metrics(cfg, info, stem)
  │
  ├── action present?          → plot_action_probabilities (bar)
  ├── action_probs present?    → plot_action_probabilities (bar)
  ├── A_matrix present?        → plot_likelihood_matrix
  ├── B_matrix present?        → plot_transition_matrix
  ├── D_matrix present?        → plot_empirical_prior
  ├── C_matrix present?        → plot_prior_preferences
  ├── q_pi present?            → plot_policy_posterior
  ├── G_epistemic + pragmatic? → plot_efe_components
  ├── qs with ≥2 timesteps?    → plot_beliefs_heatmap + plot_entropy_trajectory
  │   ├── + D_matrix?          → plot_kl_divergence_trajectory
  │   └── qs ≥3 timesteps?    → plot_belief_trajectory_animation
  ├── qpi T≥2, π≥2?           → plot_policy_posterior_heatmap
  ├── neg_efe T≥2, π≥2?       → plot_efe_trajectory + plot_neg_efe_heatmap
  ├── action ≥4, ≥2 unique?   → plot_action_frequency_donut
  ├── vfe/F size > 1?         → plot_free_energy
  └── I_matrix/I ndim ≥ 2?    → plot_reachability_matrix
```

---

## See Also

- [orchestrator_internals.md](orchestrator_internals.md) — how the post-processing pipeline triggers these functions
- [AGENTS.md](AGENTS.md) — complete trigger condition table with shape requirements
