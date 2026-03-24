# docxology/scripts

## Path resolution

- `_DOCXOLOGY = Path(__file__).resolve().parent.parent`
- `_REPO = _DOCXOLOGY.parent` (pymdp root)

[run_docxology_notebooks.py](run_docxology_notebooks.py) sets subprocess `cwd` to `_REPO` and invokes `sys.executable scripts/run_notebook_manifest.py`.

[run_docxology_orchestrations.py](run_docxology_orchestrations.py) runs subprocesses with `cwd=_DOCXOLOGY`, one entry per line from `manifests/orchestrations.txt` (paths relative to `docxology/`).

## Parent

[../AGENTS.md](../AGENTS.md)
