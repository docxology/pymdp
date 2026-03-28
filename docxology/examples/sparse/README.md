# docxology/examples/sparse

> **Sparse Tensor & JAX Optimization Benchmarks**

This directory contains validated Python scenarios that exercise the sparse tensor operations in `pymdp`. These scripts verify the numerical accuracy and memory efficiency of the JAX-backed `BCOO` (Block Compressed Sparse Row) formats used for high-dimensional state spaces.

## 🚀 Sparse Workloads

| Workload | Scenario Script | Description |
| :--- | :--- | :--- |
| **BCOO Benchmark** | `sparse_jax_benchmark.py` | Validates the performance advantage of JAX sparse matrices in large-scale navigation. |

## 📊 Output Traces
Diagnostic artifacts, including memory-profile logs and sparse-factorization visuals, are archived in:
`../../output/sparse/`

## 🤖 Agentic Contract
- **Duality Integrity**: Sparse benchmarks must verify that results from `pymdp.maths.factor_dot` are identical for both dense and BCOO inputs.
- **Memory Audits**: Ensure that the sparse representation achieves at least a 50% reduction in device memory for models with factor dimensions $> 1000$.

[Parent Examples Reference](../AGENTS.md)
