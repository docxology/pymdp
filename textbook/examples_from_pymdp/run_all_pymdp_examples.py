#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]

    # Create a local virtual environment to avoid system-level installs
    venv_dir = here / ".venv_pymdp_examples"

    def get_venv_python() -> Path:
        if not venv_dir.exists():
            try:
                subprocess.run(["uv", "venv", str(venv_dir)], check=True)
            except Exception:
                subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        posix = venv_dir / "bin" / "python"
        win = venv_dir / "Scripts" / "python.exe"
        return posix if posix.exists() else win

    vpython = str(get_venv_python())

    # Bootstrap pip in the venv and install baseline requirements and this repo
    try:
        subprocess.run([vpython, "-m", "ensurepip", "--upgrade"], check=True)
    except Exception:
        pass
    try:
        subprocess.run([vpython, "-m", "pip", "install", "--quiet", "--upgrade", "pip", "setuptools", "wheel"], check=True)
    except Exception:
        pass
    # Install this repository to make `pymdp` importable inside the venv
    try:
        subprocess.run([vpython, "-m", "pip", "install", "--quiet", "-e", str(repo_root)], check=True)
    except Exception as e:
        print(f"[WARN] Failed to install local package editable: {e}")
    # Common numeric/plotting libs used by most examples
    try:
        subprocess.run([vpython, "-m", "pip", "install", "--quiet", "numpy", "matplotlib", "seaborn"], check=True)
    except Exception as e:
        print(f"[WARN] Failed to install baseline deps: {e}")

    # 1) Generate scripts
    gen = here / "generate_examples_from_pymdp.py"
    print(f"[RUN] {gen}")
    subprocess.run([sys.executable, str(gen)], check=True)

    # 2) Discover scripts (only those we care about, matching the requested list)
    basenames = [
        "agent_demo",
        "building_up_agent_loop",
        "free_energy_calculation",
        "gridworld_tutorial_1",
        "gridworld_tutorial_2",
        "inductive_inference_example",
        "inductive_inference_gridworld",
        "model_inversion",
        "testing_large_latent_spaces",
        "tmaze_demo",
        "tmaze_learning_demo",
    ]

    scripts = [(here / f"{b}.py") for b in basenames]

    # Optional per-example extra dependencies to install if missing
    needed_modules = {
        "building_up_agent_loop": ["jax", "equinox", "opt_einsum"],
        "testing_large_latent_spaces": ["jax", "equinox", "opt_einsum", "optax"],
        "model_inversion": ["numpyro", "arviz", "seaborn", "pybefit"],
    }

    def module_exists(mod: str) -> bool:
        try:
            # Import using the current interpreter; acceptable proxy for availability
            __import__(mod)
            return True
        except Exception:
            return False

    def ensure_modules(mods):
        # Always install into local venv
        installer = [vpython, "-m", "pip", "install", "--quiet", "--upgrade"]

        to_install = [m for m in mods if not module_exists(m)]
        if to_install:
            print(f"[INSTALL] Missing modules: {to_install}")
            try:
                # Special-case JAX CPU installation (often requires jaxlib wheel source)
                if "jax" in to_install or "jaxlib" in to_install:
                    # Try pip CPU wheels
                    cmd = [vpython, "-m", "pip", "install", "--quiet", "--upgrade",
                           "jax[cpu]", "jaxlib", "-f", "https://storage.googleapis.com/jax-releases/jax_releases.html"]
                    subprocess.run(cmd, check=True)
                    # Remove jax markers from list (re-evaluate below)
                # Install remaining via chosen installer
                residual = [m for m in to_install if m not in ("jax", "jaxlib")]
                if residual:
                    subprocess.run(installer + residual, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[WARN] Failed to install some modules (code {e.returncode}). Continuing.")
            # Re-check
            still_missing = [m for m in mods if not module_exists(m)]
            if still_missing:
                print(f"[WARN] Still missing modules after install attempt: {still_missing}")
            return [m for m in mods if module_exists(m)]
        return mods
    # Include examples/agent_demo.py if it was copied (same basename)
    # The generator already ensures presence when possible

    # 3) Run each script in its own subprocess
    failures = []
    for script in scripts:
        if not script.exists():
            print(f"[SKIP] Missing: {script}")
            continue
        # Install optional extras if declared for this script
        extras = needed_modules.get(script.stem)
        if extras:
            present = ensure_modules(extras)
            # If any required module is still missing, skip this example
            if any(not module_exists(m) for m in extras):
                print(f"[SKIP] {script.stem} due to missing modules: {[m for m in extras if not module_exists(m)]}")
                continue

        # Each script executes inside its own output directory
        out_dir = here / "outputs" / script.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        harness = here / "_harness.py"
        print(f"\n[RUN] {script.name} -> {out_dir}")
        try:
            subprocess.run([vpython, str(harness), str(script), str(out_dir)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {script} exited with code {e.returncode}")
            failures.append((script, e.returncode))

    if failures:
        print("\n[SUMMARY] Some scripts failed:")
        for s, code in failures:
            print(f" - {s} (code {code})")
        return 1

    print("\n[SUMMARY] All example scripts ran (where present).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


