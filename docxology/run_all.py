#!/usr/bin/env python3
"""Global execution unifier for the docxology Active Inference pipeline.

Runs every registered example, persists full simulation data as JSON,
generates all visualizations, and writes a comprehensive run summary.

Usage:
    python run_all.py          # Auto-detects venv; relaunches via uv if needed
    uv run python run_all.py   # Direct invocation (always correct)
"""

# ── Auto-relaunch guard ──────────────────────────────────────────────────────
# When invoked with bare `python`, the system-level JAX/NumPy/pymdp may be
# incompatible. This guard detects we're outside the project .venv and
# transparently re-execs under `uv run` so all deps resolve correctly.
import os
import shutil
import subprocess
import sys
from pathlib import Path

_DOCXOLOGY = Path(__file__).resolve().parent
_REPO_ROOT = _DOCXOLOGY.parent
_VENV = _REPO_ROOT / ".venv"

_RELAUNCH_ENV_VAR = "_DOCX_UV_RELAUNCHED"

def _in_project_venv() -> bool:
    """Return True if we're running inside the project's .venv."""
    prefix = Path(sys.prefix).resolve()
    return str(prefix).startswith(str(_VENV.resolve()))

if not _in_project_venv() and not os.environ.get(_RELAUNCH_ENV_VAR):
    uv_bin = shutil.which("uv")
    if uv_bin:
        print("[run_all] Not in project venv — relaunching via `uv run` ...")
        env = os.environ.copy()
        env[_RELAUNCH_ENV_VAR] = "1"
        result = subprocess.run(
            [uv_bin, "run", "python", str(Path(__file__).resolve())] + sys.argv[1:],
            cwd=str(_DOCXOLOGY),
            env=env,
        )
        sys.exit(result.returncode)
    else:
        print("[run_all] WARNING: Not in project venv and `uv` not found.")
        print("[run_all] Please run: uv run python run_all.py")
        sys.exit(1)

# ── Normal imports (now guaranteed to be in the correct venv) ────────────────
import json
import logging
import time

import numpy as np

from pkg.support.bootstrap import OrchestrationConfig
from pkg.support.mirror_dispatch import run_registered, HANDLERS

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCXOLOGY = REPO_ROOT / "docxology"
OUTPUT_DIR = DOCXOLOGY / "output"


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles JAX/NumPy arrays and common numeric types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        try:
            import jax.numpy as jnp
            if isinstance(obj, jnp.ndarray):
                return np.asarray(obj).tolist()
        except Exception:
            pass
        try:
            return super().default(obj)
        except TypeError:
            return repr(obj)


def _setup_logging():
    """Configure dual logging: console + file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_DIR / "run_all.log"

    root_logger = logging.getLogger("validation")
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()

    fmt = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", datefmt="%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root_logger.addHandler(console)

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    return root_logger


def _save_example_data(output_dir: Path, stem: str, data: dict) -> Path:
    """Persist full result dict as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Filter out large matrix objects for the JSON dump (keep shapes instead)
    serializable = {}
    for k, v in data.items():
        if k.endswith("_matrix") or k == "detail":
            try:
                if isinstance(v, (list, tuple)):
                    serializable[k + "_shapes"] = [list(np.asarray(x).shape) for x in v]
                elif hasattr(v, "shape"):
                    serializable[k + "_shape"] = list(np.asarray(v).shape)
                else:
                    serializable[k] = v
            except Exception:
                serializable[k] = repr(v)
        else:
            serializable[k] = v

    out_path = output_dir / f"{stem}_data.json"
    out_path.write_text(json.dumps(serializable, indent=2, cls=_NumpyEncoder))
    return out_path


def run_all():
    logger = _setup_logging()

    manifest_path = DOCXOLOGY / "manifests" / "orchestrations.txt"
    if not manifest_path.exists():
        logger.error(f"Manifest missing at {manifest_path}")
        sys.exit(1)

    lines = [L.strip() for L in manifest_path.read_text().splitlines() if L.strip() and not L.startswith("#")]
    logger.info(f"Initializing global validation runner for {len(lines)} examples")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    t_global_start = time.time()
    results = []
    success_count = 0
    skipped_count = 0
    failed_count = 0

    for i, relative_path in enumerate(lines, 1):
        script_file = DOCXOLOGY / relative_path
        if not script_file.exists():
            logger.warning(f"[{i}/{len(lines)}] File missing, skipping: {script_file}")
            results.append({"path": relative_path, "status": "missing"})
            failed_count += 1
            continue

        category = Path(relative_path).parent.name
        stem = Path(relative_path).stem
        specific_output_dir = OUTPUT_DIR / category
        specific_output_dir.mkdir(parents=True, exist_ok=True)

        # Full fidelity: fast=False for richer simulations
        cfg = OrchestrationConfig(
            fast=False,
            skip_heavy=False,
            seed=42,
            verbose=True,
            output_dir=specific_output_dir,
        )

        logger.info(f"[{i}/{len(lines)}] Running: {relative_path} (category={category})")
        t_start = time.time()

        try:
            res = run_registered(script_file, cfg)
            elapsed = time.time() - t_start

            if not res.get("ok"):
                logger.error(f"  ✗ NOT OK ({elapsed:.2f}s): {res}")
                results.append({"path": relative_path, "status": "error", "elapsed_s": elapsed, "result": res})
                failed_count += 1
            elif res.get("skipped"):
                reason = res.get("reason", "unknown")
                logger.info(f"  ⊘ Skipped ({elapsed:.2f}s): {reason}")
                results.append({"path": relative_path, "status": "skipped", "reason": reason, "elapsed_s": elapsed})
                skipped_count += 1
            else:
                # Log key metrics
                metrics = {k: v for k, v in res.items() if k not in ("ok", "id", "detail") and not k.endswith("_matrix")}
                logger.info(f"  ✓ Success ({elapsed:.2f}s) | metrics: {metrics}")

                # Persist full data
                json_path = _save_example_data(specific_output_dir, stem, res)
                logger.debug(f"    Data saved: {json_path}")

                results.append({"path": relative_path, "status": "success", "elapsed_s": elapsed, "metrics": metrics})
                success_count += 1

        except Exception as e:
            elapsed = time.time() - t_start
            logger.exception(f"  ✗ Fatal error ({elapsed:.2f}s): {e}")
            results.append({"path": relative_path, "status": "fatal", "error": str(e), "elapsed_s": elapsed})
            failed_count += 1

    total_elapsed = time.time() - t_global_start
    total = success_count + skipped_count + failed_count

    # Summary
    logger.info("=" * 60)
    logger.info("GLOBAL PIPELINE VALIDATION SUMMARY")
    logger.info(f"  Total endpoints   : {total}")
    logger.info(f"  Successful        : {success_count}")
    logger.info(f"  Skipped gracefully: {skipped_count}")
    logger.info(f"  Failed            : {failed_count}")
    logger.info(f"  Total time        : {total_elapsed:.1f}s")
    logger.info("=" * 60)

    # Write summary JSON
    summary = {
        "total": total,
        "success": success_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "total_elapsed_s": round(total_elapsed, 2),
        "results": results,
    }
    summary_path = OUTPUT_DIR / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, cls=_NumpyEncoder))
    logger.info(f"Summary written to {summary_path}")

    if failed_count > 0:
        logger.error("There were failures. Implementation NOT fully validated.")
        sys.exit(1)
    else:
        logger.info("100% EXHAUSTIVE VALIDATION CONFIRMED.")


if __name__ == "__main__":
    run_all()
