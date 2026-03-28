# docxology/config

> **Orchestration Context & Rigor Settings**

This directory contains the YAML configurations that drive the docxology validation pipeline. These settings allow for precise control over execution depth, reproducibility, and logging verbosity.

## 📄 Core Configuration: `orchestration.yaml`

The primary configuration file defines the global defaults for the [thin orchestrator](../pkg/support/mirror_dispatch.py).

| Key | Type | Description |
| :--- | :--- | :--- |
| `seed` | `int` | Global PRNG seed for stochastic policy selection and environment resets. |
| `fast` | `bool` | If `true`, executes minimal timesteps (smoke test) to verify pipeline integrity. |
| `skip_heavy` | `bool` | If `true`, bypasses resource-intensive MCTS or deep-tree SI scenarios. |
| `verbose` | `bool` | Enables detailed tracing of JAX array shapes and VFE convergence during execution. |
| `output_dir` | `str` | Optional override for the base artifact directory (defaults to `../output/`). |

## 🕹️ CLI Overrides

Command-line arguments provided to [`run_docxology_orchestrations.py`](../scripts/run_docxology_orchestrations.py) always take precedence over the values defined in `orchestration.yaml`. This allows for rapid toggling between fast CI checks and rigorous local audits.

---

## 🤖 Agentic Guidance

Agents navigating this directory should:
1.  **Validate Schema**: Ensure any new keys added to `orchestration.yaml` are supported by the `OrchestrationConfig` dataclass in `pkg/support/bootstrap.py`.
2.  **Verify Loading**: Confirm that `pyyaml` is available in the environment; otherwise, the pipeline fallbacks to hardcoded defaults.

[Parent Sidecar Reference](../AGENTS.md)
