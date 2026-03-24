# docxology/manifests

## Format

- One non-empty path per line.
- Lines starting with `#` are comments (ignored).
- Blank lines ignored.
- Parser behavior matches upstream [scripts/run_notebook_manifest.py](../../scripts/run_notebook_manifest.py) `load_manifest`.

## Path roots

| Manifest | Root |
|----------|------|
| `notebooks_ci.txt`, `notebooks_nightly.txt`, `legacy_notebooks.txt`, `python_scripts.txt` | **Pymdp repository root** (parent of `docxology/`) |
| `orchestrations.txt` | **`docxology/`** directory |

## Tests

[../tests/test_manifests.py](../tests/test_manifests.py) asserts listed paths exist.

## Parent

[../AGENTS.md](../AGENTS.md)
