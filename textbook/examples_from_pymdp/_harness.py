#!/usr/bin/env python3
import atexit
import os
import runpy
import sys
from pathlib import Path


def _prepare_matplotlib(output_dir: Path):
    # Use non-interactive backend
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib
        matplotlib.use(os.environ["MPLBACKEND"], force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return  # matplotlib not used

    # Save all open figures on exit
    def _save_all_figs():
        try:
            import matplotlib.pyplot as plt  # type: ignore
            figs = list(map(plt.figure, plt.get_fignums()))
            output_dir.mkdir(parents=True, exist_ok=True)
            for idx, fig in enumerate(figs, start=1):
                out = output_dir / f"figure_{idx}.png"
                try:
                    fig.savefig(out, dpi=150, bbox_inches="tight")
                except Exception:
                    pass
        except Exception:
            pass

    atexit.register(_save_all_figs)

    # Monkey-patch plt.show to save instead of block
    try:
        _orig_show = plt.show

        def _show(*args, **kwargs):
            _save_all_figs()
            # do not block
            return None

        plt.show = _show  # type: ignore
    except Exception:
        pass


def run_script(script_path: str, output_dir: str) -> int:
    """
    Execute a python script at script_path, forcing:
    - CWD changed to output_dir (capturing any relative outputs)
    - Matplotlib figures auto-saved into output_dir
    """
    script = Path(script_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Change working directory so any outputs created are scoped per-example
    os.chdir(out_dir)

    # Prepare plotting hooks
    _prepare_matplotlib(out_dir)

    # Execute the target script in an isolated globals dict
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: _harness.py <script_path> <output_dir>")
        raise SystemExit(2)
    raise SystemExit(run_script(sys.argv[1], sys.argv[2]))



