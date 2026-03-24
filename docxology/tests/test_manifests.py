"""Manifest paths must exist; all example notebooks must be covered by a manifest."""

from __future__ import annotations

from pathlib import Path

MANIFEST_NAMES = (
    "notebooks_ci.txt",
    "notebooks_nightly.txt",
    "legacy_notebooks.txt",
    "python_scripts.txt",
    "orchestrations.txt",
)


def _load_manifest_entries(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_manifest_entries_exist(repo_root: Path, docxology_root: Path) -> None:
    manifests_dir = docxology_root / "manifests"
    for name in MANIFEST_NAMES:
        path = manifests_dir / name
        assert path.is_file(), f"Missing manifest: {path.relative_to(repo_root)}"
        for rel in _load_manifest_entries(path.read_text(encoding="utf-8")):
            # Orchestration entrypoints live under docxology/examples/ (mirrors gallery layout).
            base = docxology_root if name == "orchestrations.txt" else repo_root
            root_hint = "docxology" if name == "orchestrations.txt" else "repository"
            target = base / rel
            assert target.is_file(), f"{name}: missing path {rel} (relative to {root_hint} root)"


def test_all_example_notebooks_listed(repo_root: Path, docxology_root: Path) -> None:
    """Every notebook under examples/ appears in CI, nightly, or legacy manifest."""
    manifests_dir = docxology_root / "manifests"
    covered: set[str] = set()
    for name in ("notebooks_ci.txt", "notebooks_nightly.txt", "legacy_notebooks.txt"):
        text = (manifests_dir / name).read_text(encoding="utf-8")
        covered.update(_load_manifest_entries(text))

    notebooks = sorted(
        p.relative_to(repo_root).as_posix()
        for p in repo_root.glob("examples/**/*.ipynb")
    )
    missing = [n for n in notebooks if n not in covered]
    assert not missing, (
        "Add these to docxology/manifests (ci, nightly, or legacy):\n  "
        + "\n  ".join(missing)
    )
