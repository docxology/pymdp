# pymdp Sphinx documentation

Source for the classic Sphinx + Read the Docs style build (RST, MyST-NB notebooks under `notebooks/`).

## Build

From this directory, with dependencies installed ([requirements.txt](requirements.txt)):

```bash
make html
```

Windows:

```bash
make.bat html
```

Equivalent:

```bash
sphinx-build -M html . _build
```

Open `_build/html/index.html` in a browser.

## Dependencies

Install [requirements.txt](requirements.txt) into your environment (includes `sphinx==4.2.0`, `sphinx_rtd_theme`, `myst-nb`, and a trailing editable `.` for the local package). From the **repository root**:

```bash
uv pip install -r docs/requirements.txt
```

Or use `pip` in an activated venv.

## Static assets

[conf.py](conf.py) expects `docs/_static/pymdp_logo_2-removebg.png` for `html_logo` and `html_favicon`. Add that file if missing or adjust `conf.py`.

## MyST-NB

Notebook execution uses `jupyter_execute_notebooks = "cache"` and `jupyter_cache = "notebooks"` (cache directory under `docs/`). See [AGENTS.md](AGENTS.md).

## Docxology and `examples/`

Runnable gallery notebooks for CI live under repository **`examples/`**, indexed by **`docxology/`**. This Sphinx tree uses **`docs/notebooks/`**; content can differ. See [docxology_sidecar.rst](docxology_sidecar.rst) and [../docxology/docs/AGENTS.md](../docxology/docs/AGENTS.md).

## MkDocs

A separate site lives under [../docs-mkdocs/](../docs-mkdocs/). It is not built by this Makefile.
