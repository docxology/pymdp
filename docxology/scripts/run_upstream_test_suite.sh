#!/usr/bin/env bash
# Run pymdp upstream unit tests from the repository root (same intent as CI, local machine).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

uv sync --group test

# Match default CI matrix: parallel workers, exclude nightly-marked tests.
uv run pytest test -n 2 -m "not nightly" --cov=pymdp --cov-report=term

# Optional: uncomment to mirror CI notebook tier.
# uv run python scripts/run_notebook_manifest.py docxology/manifests/notebooks_ci.txt
