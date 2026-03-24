# pymdp docxology sidecar

Documentation, notebook manifests, and validation entrypoints for the [pymdp](https://github.com/infer-actively/pymdp) repository. All curated content lives under `docxology/` so the rest of the tree can track upstream cleanly.

## Setup

### Option A — repo root environment (recommended)

From the pymdp repository root:

```bash
uv venv .venv
source .venv/bin/activate
uv sync --group test
```

Add extras when you run heavier examples:

```bash
uv sync --group test --extra nb --extra modelfit
```

Graphviz system libraries are required for `pygraphviz` (see upstream [installation.md](../docs-mkdocs/getting-started/installation.md)).

### Option B — sidecar-only environment

```bash
cd docxology
uv sync --group test
```

This installs `inferactively-pymdp` as an editable dependency from the parent directory and pulls lightweight pytest/nbval tooling into the docxology virtualenv.

To match upstream notebook CI (Graphviz headers/libs required for `pygraphviz`, see upstream [installation.md](../docs-mkdocs/getting-started/installation.md)):

```bash
cd docxology
uv sync --group test --group notebooks
```

## Run docxology tests

From `docxology/`:

```bash
uv run pytest
```

Fast checks only (manifest paths, wrapper sanity). Full notebook execution (install `--group notebooks` first; can take tens of minutes):

```bash
uv run pytest -m slow
```

Skip slow nbval locally: `PYMDP_DOCXOLOGY_SKIP_NBVAL=1 uv run pytest -m slow` (skips the heavy test). If `mediapy` or `pygraphviz` is missing, the slow test skips with an install hint (full nbval still needs `--group notebooks` and system Graphviz).

## Run docxology Python orchestrations

From `docxology/` (paths in `manifests/orchestrations.txt` are relative to this folder):

```bash
uv run python scripts/run_docxology_orchestrations.py
```

Optional flags override `config/orchestration.yaml` (`--fast`, `--skip-heavy`, `--seed`, `-v`). Defaults load when the `test` dependency group is synced (includes `pyyaml`).

## Run notebooks via upstream nbval driver

From the **repository root** (with root or docxology env activated):

```bash
uv run python docxology/scripts/run_docxology_notebooks.py
```

Default manifest: `docxology/manifests/notebooks_ci.txt`. Pass another manifest path (repo-relative):

```bash
uv run python docxology/scripts/run_docxology_notebooks.py docxology/manifests/notebooks_nightly.txt
```

Forward flags to pytest after `--`:

```bash
uv run python docxology/scripts/run_docxology_notebooks.py -- --tb=short -v
```

Strict output matching:

```bash
uv run python docxology/scripts/run_docxology_notebooks.py --strict-output
```

Equivalent upstream command:

```bash
uv run python scripts/run_notebook_manifest.py docxology/manifests/notebooks_ci.txt
```

## Run full upstream unit tests

```bash
bash docxology/scripts/run_upstream_test_suite.sh
```

Or manually from repo root: `uv run pytest test -n 2 -m "not nightly"`.

## Documentation in this folder

| File | Purpose |
|------|---------|
| [AGENTS.md](AGENTS.md) | Layout, manifests, path rules |
| [docs/README.md](docs/README.md) | Quick links for `docxology/docs/` |
| [docs/AGENTS.md](docs/AGENTS.md) | Catalog/matrix contracts, Sphinx vs `examples/` notebooks |
| [docs/examples_catalog.md](docs/examples_catalog.md) | Every example, deps, validation pointers |
| [docs/validation_matrix.md](docs/validation_matrix.md) | Package areas → upstream `test/` modules → Sphinx |
| Subfolder `README.md` / `AGENTS.md` | `config/`, `manifests/`, `pkg/`, `scripts/`, `tests/`, `examples/` |

Run **all** docxology tests (including slow nbval): `uv sync --group test --group notebooks`, then `uv run pytest -m "slow or not slow"` (or `uv run pytest --override-ini "addopts="`).

## Drift vs MkDocs gallery

`docs-mkdocs/tutorials/notebooks.manifest` may list notebooks that are not present in every checkout. **This tree** is authoritative for paths: `docxology/manifests/*.txt` and the files under `examples/`. See the catalog for notes on upstream-only entries.

## CI note

GitHub Actions only loads workflows from the repository root `.github/workflows/`. Running docxology validation in hosted CI requires a root workflow or an external pipeline that invokes the commands above.
