# docxology/config

## Files

- **`orchestration.yaml`** — Default flags for [run_docxology_orchestrations.py](../scripts/run_docxology_orchestrations.py). CLI flags override YAML when both apply. Keys: `seed`, `fast`, `skip_heavy`, `verbose`, optional `output_dir`.

## Loading

[run_docxology_orchestrations.py](../scripts/run_docxology_orchestrations.py) reads `config/orchestration.yaml` via `yaml.safe_load` when `pyyaml` is installed; missing file or import yields empty defaults.

## Parent

[../AGENTS.md](../AGENTS.md)
