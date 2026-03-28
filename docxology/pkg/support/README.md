# docxology/pkg/support

> **The Thin Orchestrator Engine**

This directory contains the core functional modules that drive the docxology validation pipeline. These utilities bridge high-level orchestration scripts with the real `pymdp` library, providing automated data extraction, thermodynamic analysis, and publication-ready visualization.

## 🚀 Key Modules

| Module | Responsibility |
| :--- | :--- |
| **[`mirror_dispatch.py`](mirror_dispatch.py)** | **The Registry**: Maps example paths to specific handler functions. Implements the 6-step post-processing pipeline (VFE derivation, NPZ archiving). |
| **[`viz.py`](viz.py)** | **Aesthetics**: Provides 21+ headless-safe plotting functions for belief heatmaps, EFE landscapes, and policy posteriors. |
| **[`analysis.py`](analysis.py)** | **Analytics**: Derives Shannon entropy, KL-divergence, and VFE components from raw posteriors and likelihoods. |
| **[`bootstrap.py`](bootstrap.py)** | **Infrastructure**: Resolves relative repository paths and handles global JAX configuration/seeding. |
| **[`si_fixtures.py`](si_fixtures.py)** | **Benchmarks**: Hardcoded generative models for Sophisticated Inference (SI) and MCTS validation targets. |

## 🛠️ Internal Pipeline

The `mirror_dispatch.py` module implements a standardized execution flow for every registered handler:

```text
1. Execute Example -> 2. Verify Invariants -> 3. Derive H_qs -> 4. Archive Tensors (NPZ) -> 5. Generate JSON -> 6. Auto-Plot (PNG)
```

## 🤖 Agentic Guidance

- **Strict Imports**: Always import `pymdp` methods *inside* the handler functions to prevent global namespace pollution during batch execution.
- **Headless Plotting**: Ensure all plotting calls in `viz.py` respect the `matplotlib.use('Agg')` backend enforced in `bootstrap.py`.
- **Invariant Audits**: Use `checks.py` to verify that all $A$, $B$, and $D$ matrices remain properly normalized throughout the simulation.

[Parent Package Reference](../AGENTS.md)
