# docxology/examples — Active Inference Execution Harness (AGENTS.md)

## Purpose

Signposted, machine-validated execution core testing the `pymdp` library. This directory contains runnable, verified implementations using real `pymdp` functional methods.

## Zero-Mock Real Methods

All scripts inside these subdirectories are real `pymdp` configurable functional implementations. They use exact upstream methods via wrapper delegation in `pkg.support.mirror_dispatch`.

These scripts output verified configuration-driven telemetry directly to `docxology/output/`.

## Directory Layout

The taxonomy mirrors the `examples/` gallery. Every script serves as a validated, configurable runner for upstream execution.

| Directory                 | Scope                                                              |
| ------------------------- | ------------------------------------------------------------------ |
| `advanced/`               | Complex dependencies, optimization, neural encoders                |
| `api/`                    | Base model construction and agent instantiation                    |
| `envs/`                   | Standard environments (TMaze, GridWorld, Graph, Rollout)           |
| `experimental/`           | Bleeding-edge methods (Sophisticated Inference / Deep Tree Search) |
| `inductive_inference/`    | Reachability-based inference and inductive $I$ matrices            |
| `inference_and_learning/` | State inference paired with Dirichlet updating                     |
| `learning/`               | Isolated Dirichlet parameter learning ($pA, pB$)                   |
| `legacy/`                 | NumPy-era backward compatibility implementations                   |
| `model_fitting/`          | Parameter fitting and recovery algorithms                          |
| `sparse/`                 | Sparse tensor representation operations                            |
| `docxology/`              | Internal pipeline reflection tests                                 |

## Parent

[../AGENTS.md](../AGENTS.md)
