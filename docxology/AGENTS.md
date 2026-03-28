# docxology — Validation Sidecar Technical Reference (AGENTS.md)

## 0. Context & Orchestration Role

The `docxology/` directory is the **Machine-Validated Execution Core** for the `pymdp` Active Inference repository. It serves as an isolated, standalone orchestrator designed to provide independent confirmation of active inference technical details.

Rather than relying on mocks or partial test frameworks, this subdirectory integrates directly with genuine upstream elements (`pymdp/`, `test/`) to run verifiable workloads.

## 1. Zero-Mock Policy & Real Methods Enforcement

Agents working within or analyzing the `docxology/` sidecar MUST recognize that tests, scripts, and visualizations here are generated exclusively using **real `pymdp` methods**.

### Methods Directory Strategy

The `examples/` and `pkg/` directories inside `docxology/` form the execution harness. These directories import and utilize absolute, explicit methods from the core library, capturing end-to-end telemetry (via `run_all.py` and `pkg.support.mirror_dispatch`).

**Example Active Inference Real Methods executed by the Pipeline:**

- **Agent Initialization**: `pymdp.agent.Agent(A, B, C, D)`
- **Inference Steps**: `agent.infer_states(obs)`
- **Planning Horizons**: `agent.infer_policies(qs)` (computing strictly defined Epistemic and Pragmatic Information gains).
- **Control API**: Deep reachability evaluations via `pymdp.control.generate_I_matrix` or `mcts_policy_search`.

The resulting tensors (e.g., $q(s)$ matrices and $G(\pi)$ Expected Free Energy scalars) are serialized via the `run_all.py` Numpy Encoder and written out to `output/<category>_data.json` for validation matrix traceability.

## 2. Configurable Architecture Map

The role of the configurable structure within the `docxology/` folder allows precise decoupling of configuration from test execution logic:

| Path / Folder    | Engineering Purpose & Agent Role                                                                                                                                                                     |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`config/`**    | Defines `orchestration.yaml` (`[config/AGENTS.md]`). Controls validation rigor (`fast: true/false`, parameter bounds, seeds).                                                                        |
| **`manifests/`** | Text lists of targets. E.g., `orchestrations.txt` or `notebooks_ci.txt`. Excludes targets marked with `#`.                                                                                           |
| **`pkg/`**       | `pkg.support.*` logic. Provides telemetry handlers (`mirror_dispatch.py`) that introspect real `pymdp` methods running in `examples/`.                                                               |
| **`examples/`**  | The actual independent active inference workload executions (`[examples/AGENTS.md]`). Grouped exactly by mathematical taxonomy (e.g., `model_fitting/`, `inductive_inference/`).                     |
| **`docs/`**      | The operator index mapping the sidecar back to MkDocs/Sphinx. See `docs/examples_catalog.md` and `docs/validation_matrix.md`.                                                                        |
| **`scripts/`**   | CLI runners spanning `run_docxology_notebooks.py` and upstream bridging tools like `run_upstream_test_suite.sh`.                                                                                     |
| **`tests/`**     | Pytest hooks validating manifest path integrity and invoking `nbval` test passes. Confined to testing `docxology/` logic (`[tests/AGENTS.md]`).                                                      |
| **`run_all.py`** | The central JAX-safe execution environment that auto-resolves global namespaces, spins up standard `OrchestrationConfig` structs, loops the `manifests/`, and produces total verification summaries. |

## 3. Path Resolution Standards

Any agent writing code or executing scripts within the `docxology/` directories must adhere to the following strict `pathlib` resolutions to preserve ecosystem integrity:

- **Repository Root Inference**: `Path(__file__).resolve().parents[2]` (for files under `docxology/scripts/` or `docxology/tests/`).
- **Manifest File Rules**:
  - `orchestrations.txt`: Paths are resolved **relative to `docxology/`**.
  - `notebooks*.txt`: Paths are resolved **relative to the repository root** (tracking upstream examples cleanly).

## 4. `run_docxology_notebooks.py` Pipeline

Invokes upstream tests while remaining contained in the sidecar context:

```text
python scripts/run_notebook_manifest.py <manifest> [pytest_args...]
```

Arguments behavior:

- `manifest` fallback: `docxology/manifests/notebooks_ci.txt`.
- `--strict-output`: Passes `--nbval` to pytest natively rather than `--nbval-lax`.
- Forwards `sys.argv[--:]` identically to the underlying process.

## 5. Dependency Constraints

The `docxology/` environment explicitly pulls `inferactively-pymdp` as an editable installation alongside specialized analytical tools:

- `test` group: Required for YAML (`pyyaml`) and execution loops.
- `notebooks` group: Requires system graph dependencies (`pygraphviz`) and rendering (`mediapy`) to fully run verification metrics.
