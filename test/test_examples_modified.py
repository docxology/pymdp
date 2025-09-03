import os
from pathlib import Path


def test_generate_and_run_one_modified_example():
    repo = Path(__file__).resolve().parents[1]
    mod_dir = repo / "textbook" / "examples_from_pymdp_modified"
    gen = mod_dir / "generate_modified_examples.py"
    run_all = mod_dir / "run_all_modified_examples.py"

    assert gen.exists(), "generator script missing"
    assert run_all.exists(), "runner script missing"

    # Generate modified scripts
    os.system(f"python3 {gen}")

    # Ensure at least one modified script was created
    modified_scripts = list(mod_dir.glob("*_modified.py"))
    assert modified_scripts, "no modified scripts created"

    # Run a single lightweight example if present; else run all
    prefer = ["free_energy_calculation_modified.py", "agent_demo_modified.py"]
    target = None
    for name in prefer:
        cand = mod_dir / name
        if cand.exists():
            target = cand
            break
    if target is None:
        # fallback: run all but only validate outputs for first
        os.system(f"python3 {run_all}")
        target = modified_scripts[0]
    else:
        # run via harness so outputs land in outputs/<original_name>
        harness = mod_dir / "_harness.py"
        out_dir = mod_dir / "outputs" / target.stem.replace("_modified", "")
        out_dir.mkdir(parents=True, exist_ok=True)
        os.system(f"python3 {harness} {target} {out_dir}")

    # Verify outputs directory contains at least one file (figure or otherwise)
    out_dir = mod_dir / "outputs" / target.stem.replace("_modified", "")
    assert out_dir.exists(), "outputs dir missing"
    contents = list(out_dir.glob("**/*"))
    assert contents, "no outputs saved"


