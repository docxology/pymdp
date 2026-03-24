"""Docxology orchestration (mirrors upstream examples under docxology/examples/)."""

from __future__ import annotations

import sys
from pathlib import Path

for _d in Path(__file__).resolve().parents:
    if (_d / "manifests" / "orchestrations.txt").is_file():
        for _x in (_d, _d.parent):
            _s = str(_x)
            if _s not in sys.path:
                sys.path.insert(0, _s)
        break
else:
    raise RuntimeError("This file must live under docxology/examples/")

from pkg.support.bootstrap import OrchestrationConfig
from pkg.support.mirror_dispatch import run_registered
from pkg.support.report import main_cli


def run(cfg: OrchestrationConfig) -> dict:
    return run_registered(Path(__file__), cfg)


if __name__ == "__main__":
    raise SystemExit(main_cli(run))
