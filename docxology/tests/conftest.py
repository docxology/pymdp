"""Shared fixtures for docxology tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """pymdp repository root (parent of ``docxology/``)."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def docxology_root(repo_root: Path) -> Path:
    return repo_root / "docxology"
