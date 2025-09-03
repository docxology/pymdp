#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


def original_name_from_modified(stem: str) -> str:
    return stem[:-9] if stem.endswith("_modified") else stem


def module_exists(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def main() -> int:
    here = Path(__file__).resolve().parent

    # Ensure modified scripts are present
    gen = here / "generate_modified_examples.py"
    if not gen.exists():
        print(f"[ERR] Missing generator: {gen}")
        return 2
    print(f"[RUN] {gen}")
    subprocess.run([sys.executable, str(gen)], check=True)

    # Discover modified scripts
    scripts = sorted(here.glob("*_modified.py"))
    # Explicitly skip heavy/slow examples unless requested separately
    scripts = [
        s for s in scripts
        if s.name not in (
            "testing_large_latent_spaces_modified.py",
            "building_up_agent_loop_modified.py",
        )
    ]
    if not scripts:
        print("[WARN] No modified scripts found.")
        return 0

    failures = []
    harness = here / "_harness.py"
    # Optional per-example extra dependencies to gate execution
    needed_modules = {
        "building_up_agent_loop": ["jax", "equinox", "opt_einsum"],
        "testing_large_latent_spaces": ["jax", "equinox", "opt_einsum", "optax"],
        "model_inversion": ["numpyro", "arviz", "seaborn", "pybefit"],
    }
    for script in scripts:
        base = original_name_from_modified(script.stem)
        # Skip examples with missing optional dependencies
        extras = needed_modules.get(base)
        if extras and any(not module_exists(m) for m in extras):
            missing = [m for m in extras if not module_exists(m)]
            print(f"[SKIP] {script.name} due to missing modules: {missing}")
            continue
        out_dir = here / "outputs" / base
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[RUN] {script.name} -> {out_dir}")
        try:
            # Capture stdout/stderr to per-example log; enforce timeout to avoid hangs
            log_file = out_dir / "run.log"
            with log_file.open("w", encoding="utf-8") as lf:
                subprocess.run(
                    [sys.executable, str(harness), str(script), str(out_dir)],
                    check=True,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                )
        except subprocess.TimeoutExpired:
            print(f"[FAIL] {script} timed out")
            failures.append((script, 124))
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {script} exited with code {e.returncode}")
            failures.append((script, e.returncode))

    if failures:
        print("\n[SUMMARY] Some modified scripts failed:")
        for s, code in failures:
            print(f" - {s} (code {code})")
        return 1

    print("\n[SUMMARY] All modified example scripts ran (where present).")
    # Build site and open in browser
    try:
        site_builder = here / "build_site.py"
        if site_builder.exists():
            subprocess.run([sys.executable, str(site_builder)], check=True)
            # Open the generated HTML in a new browser tab
            import webbrowser
            index = here / "outputs" / "site" / "index.html"
            webbrowser.open_new_tab(index.as_uri())
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


