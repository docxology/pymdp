# Docxology-PyMDP: Real Overview

> **Zero-Mock Active Inference Validation Engine**
>
> Docxology is a validation sidecar for **[pymdp](https://github.com/infer-actively/pymdp)** — the JAX/Equinox Active Inference library. It orchestrates 32 real pymdp execution targets, derives thermodynamic diagnostics, and produces self-contained, mathematically verifiable performance reports for every run.

---

## What This Is

Docxology is **not** a fork or wrapper. It is a standalone validation layer that sits beside pymdp and exercises its full API surface — constructing real agents, running real rollouts, computing real Expected Free Energy, and archiving real tensor traces. Every function call dispatches to the authentic `pymdp.agent.Agent` class and its underlying JAX primitives.

### The Core Loop

Every example in the pipeline exercises the same fundamental Active Inference cycle:

```
obs → infer_states(obs) → q(s) → infer_policies(q(s)) → q(π) → sample_action(q(π)) → action
                                        ↑                                                   ↓
                                    update_empirical_prior(action, qs) ←───────────────────┘
```

This is the real `pymdp.agent.Agent` dispatch — FPI/MMP/VMP state inference, EFE-based policy evaluation, softmax action sampling, and Bayesian prior propagation.

---

## Architecture

```text
pymdp/                                    # Repository root
├── pymdp/                                # Core package (JAX/Equinox backend)
│   ├── agent.py                          # Agent class — central abstraction
│   ├── algos.py                          # 34 inference algorithms (FPI, MMP, VMP, HMM)
│   ├── control.py                        # 18 policy/EFE functions
│   ├── inference.py                      # 15 state inference functions
│   ├── learning.py                       # 4 Dirichlet parameter update functions
│   ├── maths.py                          # 32 mathematical primitives
│   ├── utils.py                          # 31 array construction utilities
│   ├── envs/                             # 6 environment classes + rollout engine
│   ├── planning/                         # Sophisticated Inference + MCTS
│   └── legacy/                           # NumPy-era backward compatibility
│
└── docxology/                            # ← Validation Sidecar (this project)
    ├── run_all.py                        # Global pipeline entry (32 examples)
    ├── pkg/support/                      # Orchestration engine
    │   ├── bootstrap.py                  # OrchestrationConfig + CLI parsing
    │   ├── mirror_dispatch.py            # 32 handlers + auto-viz + invariants + reporting
    │   ├── patterns.py                   # Reusable pymdp call patterns
    │   ├── viz.py                        # 21 visualization functions (headless-safe)
    │   ├── analysis.py                   # Shannon entropy, KL, VFE decomposition
    │   ├── si_fixtures.py                # SI/MCTS generative model builders
    │   ├── checks.py                     # Import/version probes (pymdp, torch, pybefit)
    │   ├── report.py                     # Structured JSON stdout + CLI runner
    │   └── path_hack.py                  # sys.path management for standalone scripts
    ├── tests/                            # 41 tests (all passing)
    ├── docs/                             # Modular documentation (this directory)
    ├── examples/                         # 32 thin runner scripts (12 categories)
    ├── manifests/                        # CI/nightly/legacy path lists
    ├── scripts/                          # CI/CD and build-time pipeline scripts
    ├── config/                           # Configuration files
    └── output/                           # Generated: JSON + NPZ + PNG + GIF + MD
```

---

## Pipeline: What Happens When You Run

```bash
cd docxology && uv run python run_all.py
```

### Execution Flow

```text
run_all.py
  │
  ├─→ Read manifests/orchestrations.txt (32 example paths)
  │
  └─→ For each example:
       │
       ├─→ mirror_dispatch.run_registered(script, cfg)
       │     │
       │     ├─→ HANDLERS[key](cfg) → info dict
       │     │     ├── Constructs real pymdp Agent(A, B, C, D)
       │     │     ├── Runs infer_states → infer_policies → sample_action
       │     │     └── Returns full diagnostic payload
       │     │
       │     └─→ _auto_plot_metrics(cfg, info, stem)
       │           ├── _verify_invariants(info)           → normalization audits
       │           ├── H_qs retroactive derivation         → Shannon entropy from qs
       │           ├── _save_native_arrays(cfg, info, stem)→ {stem}_model_trace.npz
       │           ├── JSON serialization                  → {stem}_full_data.json
       │           ├── Conditional visualization triggers  → PNG plots
       │           └── _generate_markdown_report(...)      → {stem}_execution_report.md
       │
       └─→ _save_example_data(output_dir, stem, res) → {stem}_data.json
```

### Output Per Example

Every single example produces four standard artifacts:

| Artifact | File | Content |
|---|---|---|
| Validation JSON | `{stem}_validation.json` | Handler return dict (ok, id, diagnostics) |
| Full data JSON | `{stem}_full_data.json` | Complete info dict with derived H_qs and _invariants |
| Native trace | `{stem}_model_trace.npz` | Compressed NumPy archive of ALL tensor keys |
| Execution report | `{stem}_execution_report.md` | Config, invariants, Performance Insights, embedded PNGs |

Plus conditional visualizations (up to 13 types depending on data available).

---

## Key Capabilities

### 1. Mathematical Invariant Validation

Every execution automatically audits probability assumptions:

- **Beliefs `qs`**: final posterior distribution sums to 1.0 (±1e-3)
- **Policy posterior `qpi`**: final distribution sums to 1.0
- **Likelihood `A_matrix`**: columns sum to 1.0
- **Transitions `B_matrix`**: columns sum to 1.0

Results: `{"passed": true/false, "violations": [...]}` — logged in JSON and execution reports.

### 2. Retroactive Thermodynamic Derivation

If a script returns belief sequences (`qs`) without computing entropy, the orchestrator derives it:

```python
H(q) = −Σ q(s) log q(s)    # Shannon entropy per timestep
```

The resulting `H_qs` trajectory is injected into `_full_data.json`, `_model_trace.npz`, and the Performance Insights table.

### 3. Unrestricted Tensor Serialization

The `_save_native_arrays` function iterates **all** non-dict, non-private keys in the info dict and archives them via `numpy.savez_compressed`. No whitelist, no filtering — every array that flows through the pipeline is preserved for offline analysis.

### 4. Performance Insights

The execution report extracts terminal scalar values into a Markdown table:

| Metric | Terminal Value | Mean Trajectory Value |
|---|---|---|
| Shannon Entropy H(q) | 0.6931 | 1.1773 |
| Negative Expected Free Energy −G | 1.2246 | 1.2364 |

### 5. 21 Visualization Functions

Headless-safe (`MPLBACKEND=Agg`) publication-ready plots auto-triggered by data presence:

| Category | Functions | Trigger |
|---|---|---|
| Generative model | A/B/C/D matrix heatmaps | `A_matrix` present |
| Inference | Beliefs heatmap, entropy, KL | `qs` with ≥2 timesteps |
| Policy | Posterior bar/heatmap, EFE trajectory/heatmap | `qpi`/`neg_efe` present |
| Actions | Frequency bar, donut chart | `action` present |
| Free Energy | VFE line plot | `vfe`/`F` present |
| Reachability | I matrix heatmap | `I_matrix`/`I` present |
| Animation | Belief trajectory GIF | `qs` with ≥3 timesteps |

---

## The 32 Examples

Organized across 12 categories exercising the complete pymdp API surface:

| Category | Count | pymdp Features Exercised |
|---|---|---|
| API | 1 | Agent construction, A/B/C/D matrices, one-cycle inference |
| Advanced | 3 | B_action_dependencies, FPI vs MMP comparison, torch+JAX |
| Environments | 6 | TMaze, GeneralizedTMaze, CueChaining, Graph, Grid, rollout() |
| Sophisticated Inference | 5 | si_policy_search (×3), mcts_policy_search (×2) |
| Inductive Inference | 2 | generate_I_matrix, reachability landscapes |
| Inference & Learning | 2 | FPI vs MMP, Dirichlet parameter updates |
| Model Fitting | 2 | pybefit integration, SVI recoverability |
| Sparse | 1 | JAX BCOO sparse operations |
| Legacy | 6 | NumPy-era Agent, TMazeEnv, GridWorldEnv, VFE computation |
| Docxology | 4 | Minimal cycle, learning, rollout, SI planning |

→ Full per-example details: [examples_catalog.md](examples_catalog.md)

---

## Testing

**41 tests**, all passing:

| Test File | Count | Focus |
|---|---|---|
| `test_orchestrations.py` | 32 | Every handler with `--fast --skip-heavy` |
| `test_viz.py` | 3 | Beliefs heatmap, free energy, action probs |
| `test_analysis.py` | 3 | Entropy, KL divergence, action marginalization |
| `test_manifests.py` | 2 | Manifest path existence, notebook listing |
| `test_notebook_runner_smoke.py` | 1 | Module help flag |

```bash
cd docxology && uv run pytest tests/ -v
```

---

## Configuration

| Flag | CLI | Default | Effect |
|---|---|---|---|
| `fast` | `--fast` | `False` | Shorter loops (2–3 vs 15–25 timesteps) |
| `skip_heavy` | `--skip-heavy` | `False` | Skip torch, long SI/MCTS, pybefit |
| `seed` | `--seed N` | `0` | JAX PRNG seed for reproducibility |
| `verbose` | `-v` | `False` | Enable verbose logging |
| `output_dir` | `--output-dir PATH` | `None` | Output directory for plots/data |

---

## Documentation Map

This directory contains modular documentation files, each covering a specific aspect:

| File | Purpose |
|---|---|
| **[docxology_pymdp_overview.md](docxology_pymdp_overview.md)** | This file — unified entry point and architectural overview |
| **[AGENTS.md](AGENTS.md)** | Comprehensive technical reference: pymdp package API (160+ functions), docxology modules, diagnostics schema |
| **[examples_catalog.md](examples_catalog.md)** | Per-example deep dives: all 32 examples by category with handler configs, diagnostics, viz types |
| **[validation_matrix.md](validation_matrix.md)** | Capability validation: 12 areas → tests → examples → docs → diagnostics → key APIs |
| **[README.md](README.md)** | Operator index: quick-start, pipeline at a glance, output structure, doc tree relationships |
| **[orchestrator_internals.md](orchestrator_internals.md)** | Deep-dive into mirror_dispatch.py: handler architecture, auto-plot triggers, invariant validation |
| **[visualization_reference.md](visualization_reference.md)** | Complete reference for all 21 viz functions: signatures, colormaps, trigger conditions |
| **[thermodynamics_reference.md](thermodynamics_reference.md)** | Mathematical foundations: VFE, EFE, Shannon entropy, Dirichlet learning, reachability |

---

## Dependencies

| Package | Role |
|---|---|
| `inferactively-pymdp` | Core Active Inference library (JAX/Equinox) |
| `jax` / `jaxlib` | Differentiable array computation |
| `equinox` | Pytree-based neural network modules |
| `numpy` | Array operations and NPZ serialization |
| `matplotlib` | Headless visualization (Agg backend) |
| `networkx` | Graph environment construction |
| `Pillow` | GIF animation export |
| `pybefit` | (optional) Bayesian model fitting |
| `torch` | (optional) Neural encoder integration |

---

## Quick Start

```bash
# Full 32-example pipeline
cd docxology && uv run python run_all.py

# Fast mode (shorter loops)
uv run python scripts/run_docxology_orchestrations.py --fast

# Unit tests
cd docxology && uv run pytest tests/ -v

# Upstream pymdp test suite
bash docxology/scripts/run_upstream_test_suite.sh
```

---

## See Also

- [`../AGENTS.md`](../AGENTS.md) — docxology top-level layout and contracts
- [`../README.md`](../README.md) — setup, dependency groups, full workflow
- [`../../AGENTS.md`](../../AGENTS.md) — pymdp repository-level machine-optimized index
