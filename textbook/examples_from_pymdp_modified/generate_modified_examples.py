#!/usr/bin/env python3
import re
import shutil
from pathlib import Path


HEADER = """
# Auto-generated modified copy of examples_from_pymdp
# All relative outputs (plots, files) will be saved under ./outputs/<example_name> by the harness.
""".lstrip()


def transform_filename(src: Path) -> str:
    stem = src.stem + "_modified"
    return stem + src.suffix


def ensure_harness(dst_dir: Path):
    """
    Copy the existing harness from textbook/examples_from_pymdp/_harness.py
    so we can reuse the same non-blocking figure-saving behavior.
    """
    here = Path(__file__).resolve().parent
    src_harness = here.parents[0] / "examples_from_pymdp" / "_harness.py"
    if not src_harness.exists():
        # Fallback: locate in the textbook directory
        src_harness = here.parent / "examples_from_pymdp" / "_harness.py"
    dst = dst_dir / "_harness.py"
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_harness, dst)


def main() -> int:
    src_root = Path(__file__).resolve().parent.parent / "examples_from_pymdp"
    dst_root = Path(__file__).resolve().parent

    dst_root.mkdir(parents=True, exist_ok=True)
    ensure_harness(dst_root)

    py_files = sorted(src_root.glob("*.py"))
    # Exclude harness and runner helpers; only copy example scripts
    exclude = {"_harness.py", "run_all_pymdp_examples.py", "generate_examples_from_pymdp.py"}
    for src in py_files:
        if src.name in exclude:
            continue
        dst_name = transform_filename(src)
        dst = dst_root / dst_name

        # Do not overwrite an existing modified script; preserve local edits
        if dst.exists():
            continue

        text = src.read_text(encoding="utf-8")

        # Targeted, non-breaking fixes for known incompatibilities
        if src.name == "building_up_agent_loop.py":
            # Update legacy infer_states calls to new signature
            text = re.sub(
                r"agent\.infer_states\(outcomes,\s*actions,\s*\*carry\['args'\]\)",
                "agent.infer_states(outcomes, carry['args'][0], past_actions=actions, qs_hist=carry['args'][1])",
                text,
            )
            text = re.sub(
                r"agent\.infer_states\(output\['outcomes'\],\s*output\['actions'\],\s*\*args\)",
                "agent.infer_states(output['outcomes'], args[0], past_actions=output['actions'], qs_hist=args[1])",
                text,
            )

        if src.name == "testing_large_latent_spaces.py":
            # Update legacy infer_states calls inside helper functions
            text = re.sub(
                r"agent\.infer_states\(outcomes,\s*actions,\s*\*args\)",
                "agent.infer_states(outcomes, args[0], past_actions=actions, qs_hist=args[1])",
                text,
            )
            # Remove IPython magics like %timeit (not valid in plain Python)
            text = re.sub(r"^%timeit.*$", r"# \g<0>", text, flags=re.MULTILINE)

        # Prepend a small header; otherwise keep code identical (non-breaking)
        out = HEADER + "\n" + text
        dst.write_text(out, encoding="utf-8")

    # Create outputs directory structure mirroring original examples (by example stem)
    outputs = dst_root / "outputs"
    outputs.mkdir(exist_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


