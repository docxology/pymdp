# docxology/examples/sparse — Sparse Benchmark Agentic Contract

## 0. Purpose
Machine-optimized index for the `sparse` category. These scripts serve as the definitive execution targets for validating $BCOO$ efficiency and numerical parity in high-dimensional state spaces.

## 1. Registry

| Scenario | Primary API | Key Diagnostics |
| :--- | :--- | :--- |
| `sparse_jax_benchmark.py` | `jax.experimental.sparse.BCOO` | `sparse_dot_ms`, `dense_dot_ms`, `mem_usage` |

## 2. Invariant Constraints
Agents executing these scripts MUST enforce the following numerical invariants:
- **Numerical Parity**: The result of `pymdp.maths.factor_dot(A, qs)` must be identical (within $\epsilon=1e-6$) regardless of whether $A$ is in a dense or BCOO format.
- **Memory Efficiency**: For grids larger than $50 \times 50$, the BCOO representation must utilize strictly less device memory than the corresponding dense tensor.
- **JIT Compatibility**: All sparse operations must be compatible with JAX JIT-compilation without triggering concrete-value errors.

## 3. Output Routing
All byproducts (NPZ, JSON, PNG) are routed via the [thin orchestrator](../../pkg/support/mirror_dispatch.py) to:
`docxology/output/sparse/{scenario_stem}/*`

[Parent Examples Index](../AGENTS.md)
