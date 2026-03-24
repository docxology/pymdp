"""Environment setup and CLI parsing for docxology orchestrations."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OrchestrationConfig:
    """Runtime flags for example scripts (shared across orchestrations)."""

    fast: bool
    skip_heavy: bool
    seed: int
    verbose: bool
    output_dir: Path | None

    @property
    def save_plots(self) -> bool:
        return self.output_dir is not None


def docxology_root_from_here(file: Path) -> Path:
    """Walk parents until the docxology sidecar root (manifests/orchestrations.txt)."""
    p = file.resolve()
    for parent in p.parents:
        if (parent / "manifests" / "orchestrations.txt").is_file():
            return parent
    raise RuntimeError(f"Could not locate docxology/ above {p}")


def repo_root_from_here(file: Path) -> Path:
    """Parent of ``docxology/`` (pymdp repository root)."""
    return docxology_root_from_here(file).parent


def ensure_sys_path(file: Path) -> tuple[Path, Path]:
    """Insert docxology root and pymdp repo root on ``sys.path``."""
    from pkg.support.path_hack import prepend_docxology_and_repo

    return prepend_docxology_and_repo(file)


def configure_runtime(*, matplotlib_agg: bool = True) -> None:
    """Best-effort headless plotting and deterministic JAX-friendly defaults."""
    if matplotlib_agg:
        os.environ.setdefault("MPLBACKEND", "Agg")


def parse_config(argv: list[str] | None = None) -> OrchestrationConfig:
    parser = argparse.ArgumentParser(description="docxology pymdp orchestration")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Shorter loops / smaller models where supported.",
    )
    parser.add_argument(
        "--skip-heavy",
        action="store_true",
        help="Skip optional stacks (torch, long SI/MCTS, full pybefit).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set, save figures under this directory.",
    )
    args = parser.parse_args(argv)
    return OrchestrationConfig(
        fast=args.fast,
        skip_heavy=args.skip_heavy,
        seed=args.seed,
        verbose=args.verbose,
        output_dir=args.output_dir,
    )
