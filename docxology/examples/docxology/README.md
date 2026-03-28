# docxology/examples/docxology

> **Sidecar Reflection & Rollout Benchmarks**

This directory contains validated Python scenarios that exercise the internal orchestration and diagnostic capabilities of the `docxology` sidecar itself. These scripts verify that the trace extraction, visual auto-triggering, and byproduct archiving functions remain consistent across different library versions.

## 🚀 Sidecar Workloads

| Workload | Scenario Script | Description |
| :--- | :--- | :--- |
| **Simple Rollout** | `simple_rollout.py` | A minimal, multi-timestep agent simulation used to verify the `qs` trace extraction. |
| **Diagnostic Sweep** | `all_viz_smoke_test.py` | Force-triggers all 13 visualization types to ensure headless-safe plotting stability. |

## 📊 Output Traces
Diagnostic artifacts, including the reflective traces of the sidecar's own operation, are archived in:
`../../output/docxology/`

## 🤖 Agentic Contract
- **Reflection Isolation**: Sidecar benchmarks must not interfere with the global JAX PRNG state used by other examples.
- **Artifact Verification**: Use `simple_rollout.py` to verify that the retroactive Shannon entropy derivation ($H_{qs}$) matches the theoretical values for a binary state-space.

[Parent Examples Reference](../AGENTS.md)
