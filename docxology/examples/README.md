# docxology/examples

Thin mirror scripts aligned with gallery layout under `examples/` (same suffix paths). Listed in [manifests/orchestrations.txt](../manifests/orchestrations.txt) (paths **relative to `docxology/`**).

Run all:

```bash
cd docxology && uv run python scripts/run_docxology_orchestrations.py
```

Subdirectories mirror upstream topics (`api/`, `envs/`, `legacy/`, etc.). Each script delegates to `pkg.support.mirror_dispatch` and real pymdp code.

See [../docs/examples_catalog.md](../docs/examples_catalog.md) for the authoritative gallery under repo `examples/`.
