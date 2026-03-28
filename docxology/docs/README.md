# docxology/docs — Operator Index

Technical documentation hub for the pymdp **docxology** validation sidecar. This acts as a bridge, synthesizing the theoretical concepts from the **[MkDocs Guides & Tutorials](../../docs-mkdocs/)** and the **[Sphinx API Reference](../../docs/)** against executable test pipelines.

> **Navigation Hub:** Use this sidecar documentation to trace _how_ `pymdp` capabilities are tested, while referencing `../../docs-mkdocs/` for overarching guides and `../../docs/` for formal library structures.

---

## Documentation Map

| File | Purpose |
|---|---|
| **[docxology_pymdp_overview.md](docxology_pymdp_overview.md)** | Unified entry point — architecture, pipeline flow, capabilities, quick start |
| **[AGENTS.md](AGENTS.md)** | Comprehensive technical reference: pymdp API (160+ functions), docxology modules (21 viz, 32 handlers, 4 analysis), diagnostics schema, output routing |
| **[examples_catalog.md](examples_catalog.md)** | Per-example deep dives: all 32 examples by category with handler configs, diagnostics, viz types, auto-generated artifacts |
| **[validation_matrix.md](validation_matrix.md)** | Capability validation: 12 areas → tests → examples → docs → diagnostics → key APIs → verified behaviors |
| **[orchestrator_internals.md](orchestrator_internals.md)** | Deep-dive into mirror_dispatch.py: handler patterns, post-processing pipeline, invariant validation, serialization |
| **[visualization_reference.md](visualization_reference.md)** | Complete reference for all 21 viz functions: signatures, colormaps, trigger conditions, auto-trigger flow |
| **[thermodynamics_reference.md](thermodynamics_reference.md)** | Mathematical foundations: VFE, EFE, Shannon entropy, KL divergence, Dirichlet learning, reachability, SI |
| **[README.md](README.md)** | This file — operator index, quick start, output structure, doc tree relationships |

---

## Pipeline at a Glance

**32 examples** • **30/32 full diagnostics** • **200+ output files** • **41 tests passing**

### Architecture

```
pymdp/                                    # Repository root
├── pymdp/                                # Core package (JAX backend)
│   ├── agent.py                          # Agent class (eqx.Module)
│   ├── algos.py                          # 34 inference algorithms
│   ├── control.py                        # 18 policy/EFE functions
│   ├── inference.py                      # 15 state inference functions
│   ├── learning.py                       # 4 Dirichlet update functions
│   ├── maths.py                          # 32 math primitives
│   ├── utils.py                          # 31 array utilities
│   ├── envs/                             # 6 environment classes + rollout
│   ├── planning/                         # SI tree search + MCTS
│   └── legacy/                           # NumPy-era compatibility
│
└── docxology/                            # Validation sidecar
    ├── run_all.py                        # 32-example pipeline
    ├── pkg/support/
    │   ├── bootstrap.py                  # OrchestrationConfig + CLI
    │   ├── mirror_dispatch.py            # 32 handlers + auto-viz + invariants + reporting
    │   ├── patterns.py                   # Reusable pymdp call patterns
    │   ├── viz.py                        # 21 plotting functions
    │   ├── analysis.py                   # Entropy, KL, VFE decomposition
    │   └── si_fixtures.py                # SI/MCTS model builders
    ├── tests/                            # 41 tests (all passing)
    ├── docs/                             # ← You are here
    ├── manifests/                         # CI/nightly/legacy path lists
    └── output/                           # Generated: JSON + NPZ + PNG + GIF + MD
```

### Active Inference Loop (what each example exercises)

```
          ┌──────────────────────────────────────────────────┐
          │                                                  │
obs ──→ infer_states(obs) ──→ q(s) ──→ infer_policies(q(s))│
                                            │               │
                                       q(π), −G(π)          │
                                            │               │
                              sample_action(q(π)) ──→ action│
                                                       │    │
                              update_empirical_prior(a,q) ──┘
```

---

## Quick Start

```bash
# Full 32-example pipeline (from docxology/)
cd docxology && uv run python run_all.py

# Unit tests (41 tests)
cd docxology && uv run pytest tests/ -v

# CI-tier notebooks
uv run python docxology/scripts/run_docxology_notebooks.py

# Upstream pymdp test suite
bash docxology/scripts/run_upstream_test_suite.sh
```

---

## Configuration Reference

| Flag         | CLI                 | Default | Effect                                 |
| ------------ | ------------------- | ------- | -------------------------------------- |
| `fast`       | `--fast`            | `False` | Shorter loops (2–3 vs 15–25 timesteps) |
| `skip_heavy` | `--skip-heavy`      | `False` | Skip torch, long SI/MCTS, pybefit      |
| `seed`       | `--seed N`          | `0`     | JAX PRNG seed for reproducibility      |
| `verbose`    | `-v`                | `False` | Enable verbose logging                 |
| `output_dir` | `--output-dir PATH` | `None`  | Output directory for plots/data        |

---

## Output Structure

```
output/
├── run_all.log                    # Timestamped pipeline log
├── run_summary.json               # {results: [{path, ok, elapsed_s, metrics}]}
├── api/
│   └── model_construction_tutorial/
│       ├── model_construction_validation.json
│       ├── model_construction_full_data.json     # All diagnostics + derived H_qs
│       ├── model_construction_model_trace.npz   # Native NumPy tensor archive
│       ├── model_construction_execution_report.md # Auto Markdown report
│       ├── model_construction_matrix_A.png
│       ├── model_construction_matrix_B.png
│       ├── model_construction_matrix_C.png
│       ├── model_construction_matrix_D.png
│       ├── model_construction_actions.png
│       └── model_construction_action_prob.png
├── envs/                          # 6 examples, richest output
│   └── tmaze_demo/
│       ├── tmaze_demo_validation.json
│       ├── tmaze_demo_full_data.json
│       ├── tmaze_demo_model_trace.npz
│       ├── tmaze_demo_execution_report.md
│       ├── tmaze_demo_beliefs.png
│       ├── tmaze_demo_entropy.png
│       ├── tmaze_demo_efe_traj.png
│       ├── tmaze_demo_efe_heatmap.png
│       ├── tmaze_demo_qpi_heatmap.png
│       └── tmaze_demo_beliefs_anim.gif
├── advanced/                      # complex_action, infer_states, neural_encoder
├── experimental/sophisticated_inference/  # SI×3 + MCTS×2
├── inductive_inference/           # Reachability I matrix
├── inference_and_learning/        # FPI vs MMP comparison
├── learning/                      # Dirichlet parameter updates
├── model_fitting/                 # pybefit rollout + SVI recovery
├── sparse/                        # JAX BCOO benchmark
├── legacy/                        # 6 NumPy-era examples
└── docxology/                     # 4 sidecar examples
```

---

## Diagnostics Coverage

| Handler Type | Examples                       | Keys Captured                                            |
| ------------ | ------------------------------ | -------------------------------------------------------- |
| JAX rollout  | #5–9, #20, #31                 | `qs`, `qpi`, `neg_efe`, `action`, `obs` per timestep     |
| Legacy loop  | #23–27                         | `qs`, `q_pi`, `EFE`, `VFE`, `actions`, `states` per step |
| Single-shot  | #1–4, #11–19, #22, #28–30, #32 | `beliefs`, `q_pi`, `neg_efe`                             |
| Upstream     | #10, #21                       | GIFs / SVI metrics                                       |

---

## Visualization Types (13 auto-triggered)

| Type                  | Trigger Condition             | Function                           |
| --------------------- | ----------------------------- | ---------------------------------- |
| Beliefs heatmap       | `qs` with ≥2 timesteps         | `plot_beliefs_heatmap`             |
| Entropy trajectory    | Same                          | `plot_entropy_trajectory`          |
| KL from prior         | `qs` + `D_matrix`             | `plot_kl_divergence_trajectory`    |
| EFE trajectory        | `neg_efe` with T≥2, π≥2       | `plot_efe_trajectory`              |
| Neg-EFE heatmap       | Same                          | `plot_neg_efe_heatmap`             |
| Policy posterior hmap  | `qpi` with T≥2, π≥2          | `plot_policy_posterior_heatmap`    |
| Policy posterior bar  | `q_pi` present                | `plot_policy_posterior`            |
| EFE breakdown         | `G_epistemic` + `G_pragmatic` | `plot_efe_components`              |
| Action donut          | `action` with ≥2 values       | `plot_action_frequency_donut`      |
| Belief animation      | `qs` with ≥3 timesteps         | `plot_belief_trajectory_animation` |
| Generative model      | `A_matrix` present            | `plot_likelihood_matrix` etc.      |
| VFE trajectory        | `vfe` or `F` with size > 1   | `plot_free_energy`                 |
| Reachability matrix   | `I_matrix` or `I` (ndim ≥2)   | `plot_reachability_matrix`         |

---

## Auto-Generated Artifacts (per example)

Every example automatically produces these standard outputs via the orchestrator pipeline:

| Artifact               | File                          | Description                                                     |
| ---------------------- | ----------------------------- | --------------------------------------------------------------- |
| Validation JSON        | `{stem}_validation.json`      | Handler return dict (ok, id, diagnostics)                       |
| Full data JSON         | `{stem}_full_data.json`       | Complete info dict with derived `H_qs` entropy and `_invariants`|
| Native trace archive   | `{stem}_model_trace.npz`      | Compressed NumPy of all tensor keys (unrestricted extraction)   |
| Execution report       | `{stem}_execution_report.md`  | Markdown: config, invariants, Performance Insights table, PNGs  |

### Mathematical Invariant Validation

The orchestrator automatically audits probability assumptions via `_verify_invariants(info)`:

- `qs` (beliefs): final distribution sums to 1.0 (±1e-3)
- `qpi` (policy posterior): final distribution sums to 1.0
- `A_matrix` (likelihood): columns sum to 1.0
- `B_matrix` (transitions): columns sum to 1.0

Results are logged as `{"passed": true/false, "violations": [...]}` in the JSON outputs and displayed in execution reports.

### Performance Insights

The execution report automatically extracts scalar trajectory endpoints into a Markdown table:

| Metric Key      | Display Name                    |
| --------------- | ------------------------------- |
| `H_qs`          | Shannon Entropy $H(q)$          |
| `vfe` / `F`     | Variational Free Energy $F$     |
| `neg_efe` / `G` | Negative Expected Free Energy   |
| `KL`            | KL Divergence $D_{KL}(q\|\|p)$  |

---

## Doc Tree Relationships

| Tree                                     | Role                                                    | Entry Point / Key Links                                                    |
| ---------------------------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------- |
| **[`docs/`](../../docs/)**               | Legacy Sphinx API docs (ReadTheDocs format)             | [`docs/index.rst`](../../docs/index.rst) (MyST-NB)                         |
| **[`docs-mkdocs/`](../../docs-mkdocs/)** | Modern MkDocs Site: Theory, Notebooks, and User Guides. | [`docs-mkdocs/index.md`](../../docs-mkdocs/index.md) (Material for MkDocs) |
| **[`docxology/docs/`](./)**              | Validation sidecar reference (Integration & Parity)     | `docxology/docs/README.md` (This file)                                     |
| **`docxology/output/`**                  | Generated validation artifacts (JSON/PNG/GIFs)          | Generated by `docxology/run_all.py`                                        |

---

## Key Mathematical Concepts (signposting)

Cross-reference `pymdp` logic directly back to overarching theory and documentation.

| Concept                | Formula                                  | Where Used (API)                                                                  | Theory / Guide Links                                                                  |
| ---------------------- | ---------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Generative Model**   | `p(o, s, π)`                             | `Agent(A, B, C, D)`                                                               | [📗 Model Structure](../../docs-mkdocs/guides/generative-model-structure.md)          |
| **VFE**                | `F = D_KL[q(s)‖p(s)] − E_q[log p(o\|s)]` | [`maths.calc_vfe`](../../docs/inference.rst), `inference.update_posterior_states` | [📗 Active Inference Loop](../../docs-mkdocs/guides/rollout-active-inference-loop.md) |
| **EFE**                | `G(π) = −Epistemic − Pragmatic`          | [`control.compute_neg_efe_policy`](../../docs-mkdocs/api/control.md)              | [📗 Control & Planning API](../../docs-mkdocs/api/control.md)                         |
| **Epistemic**          | `H[E_q[o]] − E_q[H[p(o\|s)]]`            | `control.compute_info_gain`                                                       | [📗 Control & Planning API](../../docs-mkdocs/api/control.md)                         |
| **Pragmatic**          | `E_q[o] · C`                             | `control.compute_expected_utility`                                                | [📗 Control & Planning API](../../docs-mkdocs/api/control.md)                         |
| **Policy posterior**   | `q(π) = σ(−G(π))`                        | `control.update_posterior_policies`                                               | [📗 Control & Planning API](../../docs-mkdocs/api/control.md)                         |
| **Dirichlet learning** | `pA ← pA + η·o⊗q(s)`                     | [`learning.update_obs_likelihood_dirichlet`](../../docs-mkdocs/api/learning.md)   | [📗 Learning API](../../docs-mkdocs/api/learning.md)                                  |
| **Reachability**       | `I(s) = P(reach goal \| s, d)`           | `control.generate_I_matrix`                                                       | [📗 Control API (`inductive`)](../../docs-mkdocs/api/control.md)                     |

---

## Parent Docs

- [`docxology/README.md`](../README.md) — setup, dependency groups, full workflow
- [`docxology/AGENTS.md`](../AGENTS.md) — layout, contracts, module architecture
