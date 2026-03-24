#!/usr/bin/env python3
"""Run all entries in ``docxology/manifests/orchestrations.txt`` sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_DOCXOLOGY = Path(__file__).resolve().parent.parent
_REPO = _DOCXOLOGY.parent


def _load_optional_yaml() -> dict:
    cfg_path = _DOCXOLOGY / "config" / "orchestration.yaml"
    if not cfg_path.is_file():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _load_manifest() -> list[str]:
    text = (_DOCXOLOGY / "manifests" / "orchestrations.txt").read_text(encoding="utf-8")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None, help="Override manifest path.")
    parser.add_argument("--fast", action="store_true", help="Pass --fast to each script.")
    parser.add_argument("--skip-heavy", action="store_true", help="Pass --skip-heavy to each script.")
    parser.add_argument("--no-skip-heavy", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    y = _load_optional_yaml()
    use_fast = args.fast or bool(y.get("fast", False))
    if args.no_skip_heavy:
        skip_heavy = False
    else:
        skip_heavy = args.skip_heavy or bool(y.get("skip_heavy", False))
    seed = args.seed if args.seed is not None else int(y.get("seed", 0))
    verbose = args.verbose or bool(y.get("verbose", False))

    manifest_path = args.manifest or (_DOCXOLOGY / "manifests" / "orchestrations.txt")
    lines = [
        s.strip()
        for s in manifest_path.read_text(encoding="utf-8").splitlines()
        if s.strip() and not s.strip().startswith("#")
    ]

    env = {**__import__("os").environ, "PYTHONPATH": f"{_DOCXOLOGY}{__import__('os').pathsep}{_REPO}"}

    failures: list[str] = []
    for rel in lines:
        script = _DOCXOLOGY / rel
        if not script.is_file():
            print(f"MISSING {rel}", file=sys.stderr)
            failures.append(rel)
            continue
            
        # Extract subcategory (e.g. "api" from "examples/api/script.py")
        parts = Path(rel).parts
        if len(parts) >= 3 and parts[0] == "examples":
            category = parts[1]
        else:
            category = "misc"
            
        out_dir = _DOCXOLOGY / "output" / category
        out_dir.mkdir(parents=True, exist_ok=True)
            
        cmd = [
            sys.executable,
            str(script),
            "--seed",
            str(seed),
            "--output-dir",
            str(out_dir),
        ]
        if use_fast:
            cmd.append("--fast")
        if skip_heavy:
            cmd.append("--skip-heavy")
        if verbose:
            cmd.append("-v")
        print("RUN", " ".join(cmd), flush=True)
        completed = subprocess.run(cmd, cwd=_REPO, env=env)
        if completed.returncode != 0:
            failures.append(rel)

    summary = {"ok": not failures, "failed": failures, "ran": len(lines)}
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
