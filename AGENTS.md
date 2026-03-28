# pymdp — Machine-Optimized Repository Reference (AGENTS.md)

## 0. AAIF 2025-2026 Standard Overview

This document is the **stringent, machine-optimized index** for the top-level `pymdp` Active Inference repository. It enforces the **Two-Layer Architecture**, explicitly bridging the legacy/upstream core (`pymdp/`, `docs/`) with the modern validation and theoretical infrastructure (`docxology/`, `docs-mkdocs/`).

> [!IMPORTANT]
> **Zero-Mock Policy Enforcement:** Agents MUST use real `pymdp` method signatures and validated JSON/visual outputs. No placeholder API logic or hallucinated frameworks are permitted.

---

## 1. Top-Level Directory Infrastructure

The repository employs a strictly defined modular layout:

- **`pymdp/`**: Core JAX/Equinox library source code and inference modules.
- **`docxology/`**: The **Validation Sidecar** encompassing testing logs, agent-generated reference files, JSON diagnostics, and the central Operator Index.
- **`examples/`**: 32 representative runnable Active Inference scenarios (Environments, SI, MCTS, Inductive).
- **`test/`**: Upstream PyTest suite ensuring numerical parity and structural coherence via `test_*.py`.
- **`scripts/`**: CI/CD and build-time pipeline shell scripts.
- **`paper/`**: Original JOSS documentation and academic scaffolding.
- **`notebooks/`**: Legacy `.ipynb` tutorials (mostly superceded by MkDocs).

### 1.1 Dual Documentation Trees

- **`docs/` (Sphinx)**: The legacy ReadTheDocs-based API generated from `.rst` files. Configured via `.readthedocs.yml` at root.
- **`docs-mkdocs/` (MkDocs)**: The modern, theoretical, and active user-guide documentation tree. Configured via `mkdocs.yml` at the repository root.

### 1.2 Build & Environment Files

- **`pyproject.toml` & `setup.cfg`**: Core package definition and strict dependency management lists.
- **`uv.lock`**: Deterministic dependency locking map managed by the `uv` ecosystem.
- **`.pre-commit-config.yaml`**: Pre-commit hooks for mandatory format compliance (black, ruff, etc.).
- **`.gitignore`**: Git path exclusion patterns.
- **`CONTRIBUTING.md`**: Human-readable guidelines for repository code submission.
- **`LICENSE`**: MIT open-source license text.
- **`nbval_sanitize.cfg`**: Sanitization patterns for continuous integration testing of Jupyter Notebooks.
- **`inferactively_pymdp.egg-info/`**: Local pip installation site packages metadata.
- **`AGENTS.md`**: (This file) Machine-readable top-level topological and operational index.

---

## 2. Core Active Inference Operations (Real Methods)

Agents interacting with or writing code for the `pymdp` codebase MUST navigate the generative loop using these authentic Equinox/JAX methods.

### Central Agent Cycle Implementation

```python
# The foundational agentic loop using real pymdp primitives.
from pymdp.agent import Agent

# Initialization requires valid generative arrays (A, B, C, D)
agent = Agent(A, B, C, D)

# 1. State Inference (Posterior update over hidden states)
# Dispatches to FPI, MMP, or VMP internally based on keyword `inference_algo`
qs, info = agent.infer_states(obs, empirical_prior=empirical_prior, return_info=True)

# 2. Policy Inference (Expected Free Energy evaluation)
# Epistemic Information Gain + Pragmatic Expected Utility + Novelty
q_pi, neg_efe = agent.infer_policies(qs)

# 3. Action Selection (Marginalization and softmax sampling)
action = agent.sample_action(q_pi, rng_key)

# 4. Temporal Advance (Prepare next-step prior likelihood p(s'|o))
empirical_prior = agent.update_empirical_prior(action, qs)
```

### Advanced Dispatch Verification

When implementing sophisticated patterns, adhere strictly to these precise endpoints:

- **Reachability (Inductive)**: `pymdp.control.generate_I_matrix(H, B, threshold, depth)`
- **Advanced Planning (SI)**: `pymdp.planning.si.si_policy_search(max_nodes, max_branching)`
- **Advanced Planning (MCTS)**: `pymdp.planning.mcts.mcts_policy_search(num_simulations, depth)`
- **Dirichlet Parameter Learning**: `pymdp.learning.update_obs_likelihood_dirichlet(pA, A_dependencies, qs, obs, lr)`
- **Fundamental Tensor Operations**: `pymdp.maths.factor_dot(A, qs)`, `pymdp.maths.stable_entropy()`, `pymdp.maths.calc_vfe()`

---

## 3. The `docxology/` Validation Sidecar

The `docxology/` directory acts as the **definitive source of operational truth** for the repository, strictly bridging the actual code execution to theoretical documentation. It orchestrates **32 real pymdp execution targets** across 12 categories, producing **200+ output files** per full run.

### 3.1 Documentation Hub (`docxology/docs/` — 8 files)

- **`docxology_pymdp_overview.md`**: Unified entry point — architecture, pipeline flow, capabilities, quick start.
- **`AGENTS.md`**: Comprehensive technical reference: pymdp API (160+ functions), docxology modules (21 viz, 32 handlers, 4 analysis), diagnostics schema, output routing.
- **`examples_catalog.md`**: Per-example deep dives: all 32 examples by category with handler configs, diagnostics, viz types, auto-generated artifacts.
- **`validation_matrix.md`**: Capability validation: 12 areas → tests → examples → docs → diagnostics → key APIs → verified behaviors.
- **`README.md`**: Operator Index — documentation map, output structure, doc tree relationships.
- **`orchestrator_internals.md`**: Deep-dive into `mirror_dispatch.py`: 4 handler patterns, 6-step post-processing pipeline, invariant validation.
- **`visualization_reference.md`**: Complete reference for all 21 viz functions: signatures, colormaps, trigger conditions, auto-trigger flow.
- **`thermodynamics_reference.md`**: Mathematical foundations: VFE, EFE, Shannon entropy, KL divergence, Dirichlet learning, reachability, SI.

### 3.2 Auto-Generated Artifacts (per example)

Every example automatically produces:

| Artifact | File Pattern | Content |
| --- | --- | --- |
| Validation JSON | `{stem}_validation.json` | Handler return dict (ok, id, diagnostics) |
| Full data JSON | `{stem}_full_data.json` | Complete info dict with derived Shannon entropy (`H_qs`) and mathematical invariant audits |
| Native trace archive | `{stem}_model_trace.npz` | Compressed NumPy archive of all tensor parameters (unrestricted extraction) |
| Execution report | `{stem}_execution_report.md` | Auto-generated Markdown with config, invariants, Performance Insights scalar table, embedded PNGs |

Plus conditional visualizations (up to 13 types depending on data available) via 21 headless-safe plotting functions.

### 3.3 Orchestrator Pipeline

```text
run_all.py → HANDLERS[key](cfg) → info dict
  → _verify_invariants(info)            → normalization audits
  → H_qs retroactive derivation         → Shannon entropy from qs
  → _save_native_arrays(cfg, info, stem) → model_trace.npz
  → _auto_plot_metrics(cfg, info, stem)  → conditional PNG plots
  → _generate_markdown_report(...)       → execution_report.md
```

> [!WARNING]  
> **Rule of Parity:** Any modification to the `pymdp/` source codebase MUST be simultaneously documented in the `docs-mkdocs/` API guides AND cross-referenced or demonstrated within the `docxology/` validation sidecar. Code without equivalent validation matrices is considered incomplete.
