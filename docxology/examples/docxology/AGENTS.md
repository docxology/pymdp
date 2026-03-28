# docxology/examples/docxology — Sidecar Reflection Agentic Contract

## 0. Purpose
Machine-optimized index for the `docxology` category. These scripts serve as the definitive execution targets for validating the sidecar's own orchestration and diagnostic harvesting logic.

## 1. Reflection Registry

| Scenario | Primary API | Key Diagnostics |
| :--- | :--- | :--- |
| `simple_rollout.py` | `docxology.pkg.support.mirror_dispatch` | `qs_trace`, `H_qs_derivation` |
| `all_viz_smoke_test.py` | `docxology.pkg.support.viz` | `trigger_count`, `plot_success` |

## 2. Invariant Constraints
Agents executing these scripts MUST enforce the following numerical invariants:
- **Trace Integrity**: The `qs_trace` extracted by `mirror_dispatch` must exactly match the list of `qs` objects returned by the underlying `Agent.infer_states` calls.
- **Auto-Trigger Logic**: In `all_viz_smoke_test.py`, at least 10 distinct visualization files must be generated in the output directory.
- **Entropy Parity**: The derived Shannon entropy $H_{qs}$ must handle `0.0` probability values using `pymdp.maths.stable_log` to avoid `NaN` artifacts.

## 3. Output Routing
All byproducts (NPZ, JSON, PNG) are routed via the [thin orchestrator](../../pkg/support/mirror_dispatch.py) to:
`docxology/output/docxology/{scenario_stem}/*`

[Parent Examples Index](../AGENTS.md)
