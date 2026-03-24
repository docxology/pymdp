# docs/ — Sphinx technical reference

## Layout

| Path | Role |
|------|------|
| `conf.py` | Extensions, theme, `sys.path` to parent package, MyST-NB, static paths |
| `index.rst` | Master toctree |
| `*.rst` | Module and topic pages |
| `algos/` | FPI / MMP sub-pages (JAX `pymdp.algos`) |
| `notebooks/*.ipynb` | MyST-NB sources (also listed in `index.rst`) |
| `Makefile`, `make.bat` | `sphinx-build` wrappers |
| `requirements.txt` | Pinned Sphinx stack + editable install `.` |
| `_static/` | Logo and static assets referenced by `html_logo` / `html_static_path` |
| `_build/` | Generated output (gitignored typical) |

## Extensions (`conf.py`)

- `sphinx.ext.autodoc`, `doctest`, `coverage`, `napoleon`, `autosummary`
- `myst_nb` — `.ipynb` as sources; `source_suffix` maps `.ipynb` to `myst-nb`

## MyST-NB

- `jupyter_execute_notebooks = "cache"` — reuse executed outputs when possible
- `jupyter_cache = "notebooks"` — cache subdirectory name under `docs/`

## Version

`version` / `release` in `conf.py` should stay aligned with root [pyproject.toml](../pyproject.toml) `[project].version` when bumping releases.

## Cross-links to docxology

Human-oriented indexes: [../docxology/docs/README.md](../docxology/docs/README.md). Sphinx page: [docxology_sidecar.rst](docxology_sidecar.rst).

## Parent repository

Pymdp package root is the parent of `docs/`; `conf.py` sets `sys.path.insert(0, os.path.abspath('..'))` for autodoc.
