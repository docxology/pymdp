"""Insert ``docxology/`` and pymdp repo root onto ``sys.path`` for standalone scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def prepend_docxology_and_repo(script_file: Path) -> tuple[Path, Path]:
    """Return ``(docxology_root, repo_root)`` after updating ``sys.path``."""
    doc = script_file.resolve()
    for parent in doc.parents:
        if (parent / "manifests" / "orchestrations.txt").is_file():
            repo = parent.parent
            for p in (parent, repo):
                s = str(p)
                if s not in sys.path:
                    sys.path.insert(0, s)
            return parent, repo
    raise RuntimeError(f"No docxology/ ancestor of {script_file}")
