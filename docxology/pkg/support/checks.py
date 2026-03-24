"""Import and version probes for pymdp and optional stacks."""

from __future__ import annotations

from typing import Any


def pymdp_probe() -> dict[str, Any]:
    import pymdp

    import jax

    return {
        "pymdp_file": getattr(pymdp, "__file__", None),
        "jax_version": jax.__version__,
    }


def try_import_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def try_import_pybefit() -> bool:
    try:
        import pybefit  # noqa: F401

        return True
    except Exception:
        return False


def try_import_pygraphviz() -> bool:
    try:
        import pygraphviz  # noqa: F401

        return True
    except Exception:
        return False
