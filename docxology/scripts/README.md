# docxology/scripts

Entrypoints for notebooks, orchestrations, and upstream pytest.

| Script | Description |
|--------|-------------|
| [run_docxology_notebooks.py](run_docxology_notebooks.py) | Chdir to repo root; run `scripts/run_notebook_manifest.py` on a manifest |
| [run_docxology_orchestrations.py](run_docxology_orchestrations.py) | Run every path in `manifests/orchestrations.txt` from `docxology/` |
| [run_upstream_test_suite.sh](run_upstream_test_suite.sh) | `uv run pytest test …` from repo root |

See [../README.md](../README.md) for commands.
