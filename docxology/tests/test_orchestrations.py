"""Run docxology example orchestrations as subprocesses (real pymdp execution)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_manifests import _load_manifest_entries


def _orch_scripts(docxology_root: Path) -> list[Path]:
    text = (docxology_root / "manifests" / "orchestrations.txt").read_text(encoding="utf-8")
    return [docxology_root / rel for rel in _load_manifest_entries(text)]


@pytest.mark.parametrize(
    "script_path",
    _orch_scripts(Path(__file__).resolve().parents[1]),
    ids=lambda p: p.relative_to(p.parents[1]).as_posix(),
)
def test_orchestration_script_runs(script_path: Path, repo_root: Path) -> None:
    docxology = repo_root / "docxology"
    assert script_path.is_file(), script_path
    env = {**os.environ, "PYTHONPATH": f"{docxology}{os.pathsep}{repo_root}"}
    python_exe = docxology / ".venv" / "bin" / "python"
    
    cmd = [
        str(python_exe),
        str(script_path),
        "--fast",
        "--skip-heavy",
        "--seed",
        "0",
    ]
    completed = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{script_path.name} failed:\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    lines = [ln for ln in completed.stdout.splitlines() if ln.strip().startswith("{")]
    assert lines, f"no JSON line in output: {completed.stdout!r}"
    payload = json.loads(lines[-1])
    assert payload.get("ok") is not False, payload
