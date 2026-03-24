#!/usr/bin/env python3
"""Invoke upstream `scripts/run_notebook_manifest.py` on a docxology manifest."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_DEFAULT_MANIFEST = "docxology/manifests/notebooks_ci.txt"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    argv = sys.argv[1:]
    if "--" in argv:
        split_at = argv.index("--")
        main_argv = argv[:split_at]
        pytest_args = argv[split_at + 1 :]
    else:
        main_argv = argv
        pytest_args = []

    parser = argparse.ArgumentParser(
        description="Run nbval on notebooks listed in a docxology manifest (repo-relative paths)."
    )
    parser.add_argument(
        "--strict-output",
        action="store_true",
        help="Use strict nbval output matching (upstream --nbval instead of --nbval-lax).",
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        default=_DEFAULT_MANIFEST,
        help=f"Manifest path relative to repository root (default: {_DEFAULT_MANIFEST}).",
    )
    args = parser.parse_args(main_argv)

    root = _repo_root()
    driver = root / "scripts" / "run_notebook_manifest.py"
    if not driver.is_file():
        raise SystemExit(f"Upstream notebook driver not found: {driver}")

    manifest_path = root / args.manifest
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    cmd = [
        sys.executable,
        str(driver),
        args.manifest,
    ]
    if args.strict_output:
        cmd.append("--strict-output")
    cmd.extend(pytest_args)

    completed = subprocess.run(cmd, cwd=root)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
