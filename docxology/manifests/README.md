# docxology/manifests

Text manifests consumed by docxology scripts and tests.

| File | Path root | Purpose |
|------|-----------|---------|
| [notebooks_ci.txt](notebooks_ci.txt) | pymdp repo root | CI-tier notebooks for nbval |
| [notebooks_nightly.txt](notebooks_nightly.txt) | repo root | Heavier notebooks |
| [legacy_notebooks.txt](legacy_notebooks.txt) | repo root | Legacy / NumPy-era notebooks |
| [python_scripts.txt](python_scripts.txt) | repo root | Runnable `examples/*.py` |
| [orchestrations.txt](orchestrations.txt) | **`docxology/`** | Mirror scripts under `docxology/examples/` |

Format: one path per line; `#` comments and blank lines ignored (see [AGENTS.md](AGENTS.md)).
