"""Structured stdout for orchestration runners and tests."""

from __future__ import annotations

import json
from typing import Any, Callable

from pkg.support.bootstrap import OrchestrationConfig, configure_runtime, parse_config


def emit_result(payload: dict[str, Any]) -> None:
    """Print a single JSON object line for machine-readable checks."""
    print(json.dumps(payload, sort_keys=True, default=str), flush=True)


def main_cli(run: Callable[[OrchestrationConfig], dict[str, Any]], argv: list[str] | None = None) -> int:
    import sys
    from pathlib import Path
    
    configure_runtime()
    cfg = parse_config(argv)
    try:
        out = run(cfg)
        ok = bool(out.get("ok", True))
        
        if cfg.output_dir:
            script_name = Path(sys.argv[0]).stem
            out_file = cfg.output_dir / f"{script_name}_validation.json"
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
            
        emit_result(out)
        return 0 if ok else 1
    except Exception as exc:  # noqa: BLE001 — surface failures to runner
        emit_result({"ok": False, "error": type(exc).__name__, "message": str(exc)})
        if cfg.verbose:
            raise
        return 1
