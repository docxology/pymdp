# docxology — technical reference

## Purpose

Sidecar for pymdp: manifests of runnable examples, scripts to invoke upstream `scripts/run_notebook_manifest.py`, documentation mapping examples to upstream tests, and pytest checks that stay confined to `docxology/`.

## Directory layout

| Path | Description |
|------|-------------|
| `config/orchestration.yaml` | Optional defaults for `run_docxology_orchestrations.py` ([config/AGENTS.md](config/AGENTS.md)) |
| `manifests/notebooks_ci.txt` | Notebook paths (repo-relative) aligned with upstream `test/notebooks/ci_notebooks.txt` for this tree |
| `manifests/notebooks_nightly.txt` | Longer or heavier notebooks present here; comments document upstream-only paths missing locally |
| `manifests/legacy_notebooks.txt` | NumPy / legacy-era notebooks under `examples/legacy/` |
| `manifests/python_scripts.txt` | Runnable Python examples under repo `examples/` (repo-relative paths) |
| `manifests/orchestrations.txt` | Docxology mirror scripts under `docxology/examples/` (paths relative to `docxology/`) |
| `pkg/` | `pkg.support.*` helpers for mirror scripts ([pkg/AGENTS.md](pkg/AGENTS.md)) |
| `examples/` | Mirror runners ([examples/AGENTS.md](examples/AGENTS.md)) |
| `scripts/run_docxology_notebooks.py` | Resolves repo root, delegates to `scripts/run_notebook_manifest.py` |
| `scripts/run_docxology_orchestrations.py` | Runs each line of `orchestrations.txt` under `docxology/` |
| `scripts/run_upstream_test_suite.sh` | Shell wrapper for `uv run pytest test …` from repo root |
| `tests/` | Pytest: manifest existence, optional slow nbval ([tests/AGENTS.md](tests/AGENTS.md)) |
| `docs/README.md`, `docs/AGENTS.md` | Operator + technical index for this folder |
| `docs/examples_catalog.md` | Operator-oriented example index |
| `docs/validation_matrix.md` | Capability → `test/test_*.py` → examples → Sphinx |
| `output/` | Generated local assets (optional; [output/README.md](output/README.md)) |

## Path resolution

- Repository root = `Path(__file__).resolve().parents[2]` for files in `docxology/scripts/`.
- Repository root = `Path(__file__).resolve().parents[2]` for files in `docxology/tests/` (`parents[0]` = `tests`, `parents[1]` = `docxology`, `parents[2]` = pymdp root).

Path roots:

- **Notebooks** (`notebooks_*.txt`, `legacy_notebooks.txt`), **`python_scripts.txt`**: repo-relative (under pymdp root).
- **`orchestrations.txt`**: relative to **`docxology/`** (files under `docxology/examples/…`, same `examples/…` suffix as the gallery for alignment).

## Manifest format

- One path per line (repo-relative for notebooks and `python_scripts.txt`; `docxology/`-relative for `orchestrations.txt`).
- Lines starting with `#` are ignored (same as upstream `run_notebook_manifest.load_manifest`).
- Blank lines ignored.

## `run_docxology_notebooks.py`

Invokes:

```text
python scripts/run_notebook_manifest.py <manifest> [pytest_args...]
```

with `cwd` = repository root. Uses the current interpreter (`sys.executable`).

Arguments:

- Optional `manifest` (default `docxology/manifests/notebooks_ci.txt`).
- `--strict-output` → passes `--nbval` instead of `--nbval-lax` on the underlying pytest invocation.
- After `--`, remaining tokens forwarded to pytest (e.g. `-v`, `--tb=short`).

## Pytest markers (docxology)

| Marker | Meaning |
|--------|---------|
| `slow` | Runs nbval over the CI notebook manifest (long) |

Upstream markers such as `nightly` apply only when running `pytest` from the repo root against `test/`.

## Dependencies

`docxology/pyproject.toml` pins `inferactively-pymdp` from `..` (editable).

| Group | Purpose |
|-------|---------|
| `test` | `pytest`, `nbval`, `ipykernel`, `pyyaml` — manifest tests, `--help` smoke, `config/orchestration.yaml` defaults |
| `notebooks` | `jupyter`, `mediapy`, `pygraphviz`, `pybefit` — full nbval on CI-tier notebooks (needs Graphviz system packages) |

`pybefit` source matches root `[tool.uv.sources]`.

## Related upstream files

- `scripts/run_notebook_manifest.py` — nbval driver
- `test/notebooks/ci_notebooks.txt` — upstream CI manifest
- `test/notebooks/nightly_notebooks.txt` — upstream nightly manifest
- `pyproject.toml` — root package and `[dependency-groups] test`
