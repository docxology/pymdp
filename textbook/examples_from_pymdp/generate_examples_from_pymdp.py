#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path


def read_notebook_code_cells(nb_path: Path) -> str:
    """
    Return a string consisting of the exact concatenation of all code cell sources
    from the notebook at nb_path, preserving line contents and order.
    No headers or extra lines are added.
    """
    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    content_parts = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src_text = "".join(cell.get("source", []))
            if not content_parts:
                content_parts.append(src_text)
            else:
                # Ensure there is exactly one separating newline between code cells
                prev = content_parts[-1]
                if prev and not prev.endswith("\n"):
                    content_parts[-1] = prev + "\n"
                content_parts.append(src_text)

    content = "".join(content_parts)
    # Ensure the final content ends with a newline for POSIX text file convention
    if content and not content.endswith("\n"):
        content += "\n"
    return content


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    examples_dir = repo_root / "examples"
    out_dir = Path(__file__).resolve().parent

    ensure_dir(out_dir)

    # Explicit source list from the user request
    sources = [
        "agent_demo.ipynb",
        "agent_demo.py",
        "building_up_agent_loop.ipynb",
        "free_energy_calculation.ipynb",
        "gridworld_tutorial_1.ipynb",
        "gridworld_tutorial_2.ipynb",
        "inductive_inference_example.ipynb",
        "inductive_inference_gridworld.ipynb",
        "model_inversion.ipynb",
        "testing_large_latent_spaces.ipynb",
        "tmaze_demo.ipynb",
        "tmaze_learning_demo.ipynb",
    ]

    # Strategy:
    # - If a .py of the same basename exists in examples/, copy it as-is.
    # - Else if a .ipynb exists, extract code cells to .py with identical basename.
    # This avoids duplicate name collisions (e.g., agent_demo.ipynb vs agent_demo.py).

    written = []
    skipped_notebooks = []
    for src in sources:
        src_path = examples_dir / src
        if not src_path.exists():
            print(f"[WARN] Source not found: {src_path}")
            continue

        base = src_path.stem  # without suffix
        py_src = examples_dir / f"{base}.py"
        ipynb_src = examples_dir / f"{base}.ipynb"
        out_py = out_dir / f"{base}.py"

        if py_src.exists():
            # Copy .py exactly
            data = py_src.read_bytes()
            out_py.write_bytes(data)
            print(f"[COPY] {py_src} -> {out_py}")
            written.append(out_py)
            # If notebook also exists, we consider it redundant to avoid name collision
            if ipynb_src.exists() and src_path.suffix == ".ipynb":
                skipped_notebooks.append(ipynb_src)
        elif ipynb_src.exists():
            code = read_notebook_code_cells(ipynb_src)
            out_py.write_text(code, encoding="utf-8")
            print(f"[NB->PY] {ipynb_src} -> {out_py}")
            written.append(out_py)
        else:
            print(f"[SKIP] No convertible source for {base}")

    if skipped_notebooks:
        print("[INFO] Skipped notebooks with same basename as existing .py to avoid collisions:")
        for p in skipped_notebooks:
            print(f"       - {p}")

    print(f"[DONE] Wrote {len(written)} scripts to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


