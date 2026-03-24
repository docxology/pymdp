# docxology/tests

Pytest suite for the sidecar (confined to `docxology/`).

| File | Role |
|------|------|
| [conftest.py](conftest.py) | Shared fixtures |
| [test_manifests.py](test_manifests.py) | Manifest path existence |
| [test_orchestrations.py](test_orchestrations.py) | Orchestration scripts (`--fast --skip-heavy`) |
| [test_notebook_runner_smoke.py](test_notebook_runner_smoke.py) | Wrapper `--help` / smoke |

Run from `docxology/`: `uv run pytest`. Slow nbval: `uv run pytest -m slow`.
