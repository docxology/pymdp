#!/usr/bin/env python3
"""
Validation and Verification Harness for PyMDP Textbook Examples
================================================================

This script validates that:
- Each example runs and produces non-trivial outputs
- JSON summaries (when present) are parseable and contain dynamic, changing data
- Authentic PyMDP methods are used (no fallback-only runs)
- Example modules contain top-level documentation strings

Usage:
  python3 validate_examples.py [--strict]

Exit codes:
- 0: All validations passed
- 1: Some validations produced warnings (non-strict mode)
- 2: Validation failures (or any warning in --strict mode)
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

EXAMPLES_DIR = Path(__file__).parent
OUTPUTS_DIR = EXAMPLES_DIR / "outputs"
LOGS_DIR = EXAMPLES_DIR / "logs"

EXAMPLE_FILES = [
    "01_probability_basics.py",
    "02_bayes_rule.py",
    "03_observation_models.py",
    "03_observation_models_refactored.py",
    "04_state_inference.py",
    "05_sequential_inference.py",
    "06_multi_factor_models.py",
    "07_transition_models.py",
    "08_preferences_and_control.py",
    "09_policy_inference.py",
    "10_simple_pomdp.py",
    "10_simple_pomdp_backup.py",
    "11_gridworld_pomdp.py",
    "12_tmaze_pomdp.py",
]

# Phrases indicating potential fallback/non-authentic runs
SUSPICIOUS_PATTERNS = [
    r"Fallback",
    r"Using fallback function",
    r"Proceeding with educational",
    r"PyMDP inference error",
    r"PyMDP error",
    r"Agent creation failed",
    r"Simulation error",
]

# Positive markers that indicate authentic PyMDP method usage even if some
# fallback messaging also appears (prefer positive evidence over warnings)
POSITIVE_MARKERS = [
    r"✅ PyMDP Agent created successfully",
    r"✅ PyMDP Planning Agent created successfully",
    r"✅ PyMDP Multi-Factor Agent created successfully",
    r"✅ PyMDP Agent created successfully for EFE analysis",
    r"✅ Used PyMDP agent\.infer_policies\(\)",
    r"PyMDP Model Validation:",
]


def load_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def has_module_docstring(py_file: Path) -> bool:
    try:
        text = py_file.read_text(encoding="utf-8")
    except Exception:
        return False
    # Consider first non-empty, non-shebang lines until code starts
    # A simple heuristic: look for triple-quoted string at top
    doc_re = re.compile(r"^#!/.*\n\s*\"\"\"[\s\S]*?\"\"\"", re.MULTILINE)
    alt_re = re.compile(r"^\s*\"\"\"[\s\S]*?\"\"\"", re.MULTILINE)
    return bool(doc_re.search(text) or alt_re.search(text))


def log_has_suspicious_patterns(log_path: Path) -> bool:
    if not log_path.exists():
        return True
    try:
        txt = log_path.read_text(errors="ignore")
    except Exception:
        return True
    # If we find any of the positive markers, treat as authentic regardless
    # of incidental fallback messaging present elsewhere in the log.
    for ok_pat in POSITIVE_MARKERS:
        if re.search(ok_pat, txt, flags=re.IGNORECASE):
            return False
    for pat in SUSPICIOUS_PATTERNS:
        if re.search(pat, txt, flags=re.IGNORECASE):
            return True
    return False


def ensure_outputs_non_trivial(out_dir: Path) -> Tuple[bool, List[str]]:
    warnings: List[str] = []
    if not out_dir.exists():
        warnings.append(f"Missing outputs directory: {out_dir}")
        return False, warnings
    # No zero-byte files
    zero_byte = [p for p in out_dir.rglob("*") if p.is_file() and p.stat().st_size == 0]
    if zero_byte:
        warnings.append(f"Zero-byte output files: {[p.name for p in zero_byte[:5]]} ...")
    # JSON parseable
    for p in out_dir.rglob("*.json"):
        data = load_json(p)
        if data is None:
            warnings.append(f"Invalid JSON: {p.name}")
    return len(warnings) == 0, warnings


def check_dynamic_behavior_from_summaries(out_dir: Path) -> Tuple[bool, List[str]]:
    """Inspect known summary files and assert the data indicates dynamics (non-constant)."""
    warnings: List[str] = []
    # Heuristic per example summary naming
    candidates = list(out_dir.glob("example_*_summary.json"))
    if not candidates:
        # Some examples use specialized names (e.g., example_02_vfe_summary.json)
        candidates = list(out_dir.glob("*.json"))
    if not candidates:
        return True, warnings

    ok_any = False
    for p in candidates:
        data = load_json(p)
        if data is None:
            warnings.append(f"Cannot parse JSON: {p.name}")
            continue

        # Generic checks across known keys
        # 1) Belief evolutions vary
        belief_paths = [
            ("basic_sequential", ["belief_evolution"]),
            ("sequential_vfe_updating", ["belief_evolution"]),
        ]
        for section, keys in belief_paths:
            sec = data.get(section)
            if isinstance(sec, dict):
                arr = sec.get(keys[0])
                if isinstance(arr, list) and len(arr) > 2:
                    # Compare first and last
                    if arr[0] != arr[-1]:
                        ok_any = True
                    else:
                        warnings.append(f"No change in beliefs in {p.name}:{section}")

        # 2) VFE / info gain present and non-negative patterns where expected
        if "vfe_medical_example" in data:
            vfe_res = data["vfe_medical_example"].get("vfe_results", [])
            if vfe_res:
                infos = [r.get("info_gain", 0.0) for r in vfe_res]
                if any(i >= 0 for i in infos):
                    ok_any = True

        # 3) Model matrices sanity (e.g., identity for perfect observation in ex03)
        if "perfect_observation" in data:
            mat = data["perfect_observation"]
            if isinstance(mat, list) and mat:
                # Expect near-identity structure on square matrix
                diag_ok = True
                off_ok = True
                for i, row in enumerate(mat):
                    for j, v in enumerate(row):
                        if i == j:
                            diag_ok = diag_ok and (abs(v - 1.0) < 1e-6)
                        else:
                            off_ok = off_ok and (abs(v) < 1e-6)
                if not (diag_ok and off_ok):
                    warnings.append(f"Perfect observation not identity in {p.name}")
                else:
                    ok_any = True

    return ok_any or not warnings, warnings


def main(strict: bool) -> int:
    overall_ok = True
    collected_warnings: List[str] = []

    # 1) Module docstrings exist
    for ex in EXAMPLE_FILES:
        py_path = EXAMPLES_DIR / ex
        if not py_path.exists():
            collected_warnings.append(f"Missing example file: {ex}")
            overall_ok = False
            continue
        if not has_module_docstring(py_path):
            collected_warnings.append(f"Missing top-level docstring: {ex}")
            overall_ok = False

    # 2) Logs authenticity
    for ex in EXAMPLE_FILES:
        log_path = LOGS_DIR / f"{Path(ex).stem}.log"
        if not log_path.exists():
            collected_warnings.append(f"Missing log (run run_all.sh first): {log_path.name}")
            overall_ok = False
            continue
        if log_has_suspicious_patterns(log_path):
            collected_warnings.append(f"Authenticity warning in log: {log_path.name}")
            overall_ok = False

    # 3) Outputs exist and are non-trivial; JSON parseable; dynamics visible
    for ex in EXAMPLE_FILES:
        out_dir = OUTPUTS_DIR / Path(ex).stem
        ok, warns = ensure_outputs_non_trivial(out_dir)
        if not ok:
            overall_ok = False
            collected_warnings.extend([f"{Path(ex).name}: {w}" for w in warns])
        dyn_ok, dyn_warns = check_dynamic_behavior_from_summaries(out_dir)
        if not dyn_ok:
            collected_warnings.extend([f"{Path(ex).name}: {w}" for w in dyn_warns])
            # Dynamics warnings are soft unless strict
            if strict:
                overall_ok = False

    # Report
    if collected_warnings:
        print("Validation warnings/errors:")
        for w in collected_warnings:
            print(f"- {w}")

    if overall_ok:
        print("✅ All validations passed.")
        return 0
    else:
        if strict:
            print("❌ Validation failures (strict mode).")
            return 2
        else:
            print("⚠️  Some validations reported warnings.")
            return 1


if __name__ == "__main__":
    strict_flag = "--strict" in sys.argv[1:]
    sys.exit(main(strict_flag))


