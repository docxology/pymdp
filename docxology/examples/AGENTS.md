# docxology/examples

## Purpose

Runnable **mirrors** of gallery examples: same conceptual coverage as `examples/**/*.py` where listed, but rooted under `docxology/` for isolated orchestration and pytest ([test_orchestrations.py](../tests/test_orchestrations.py)).

## Layout

Directory names align with pymdp **`examples/`** (e.g. `docxology/examples/envs/tmaze_demo.py` ↔ `examples/envs/tmaze_demo.py`). Implementation imports pymdp via `pkg.support`.

## Manifest

[../manifests/orchestrations.txt](../manifests/orchestrations.txt) — one `docxology/`-relative path per line.

## Parent

[../AGENTS.md](../AGENTS.md)
