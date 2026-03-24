"""Smoke tests for the docxology notebook runner (optional full nbval: slow)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_run_docxology_notebooks_module_help(repo_root: Path) -> None:
    script = repo_root / "docxology" / "scripts" / "run_docxology_notebooks.py"
    completed = subprocess.run(
        [sys.executable, str(script), "-h"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _skip_nbval_if_notebook_stack_incomplete() -> None:
    """CI-tier notebooks import mediapy and pygraphviz; avoid a long failing run without them."""
    try:
        import mediapy  # noqa: F401
    except ImportError:
        pytest.skip(
            "mediapy not installed; sync docxology with: uv sync --group test --group notebooks"
        )
    try:
        import pygraphviz  # noqa: F401
    except ImportError:
        pytest.skip(
            "pygraphviz not installed (needs system Graphviz); "
            "uv sync --group test --group notebooks — see docs-mkdocs/getting-started/installation.md"
        )


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_ci_notebooks_nbval(repo_root: Path) -> None:
    if os.environ.get("PYMDP_DOCXOLOGY_SKIP_NBVAL", "").lower() in ("1", "true", "yes"):
        pytest.skip("PYMDP_DOCXOLOGY_SKIP_NBVAL set")

    _skip_nbval_if_notebook_stack_incomplete()

    driver = repo_root / "scripts" / "run_notebook_manifest.py"
    manifest = "docxology/manifests/notebooks_ci.txt"
    completed = subprocess.run(
        [sys.executable, str(driver), manifest],
        cwd=repo_root,
        check=False,
    )
    assert completed.returncode == 0
