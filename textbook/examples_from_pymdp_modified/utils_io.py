from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_npy(out_dir: Path, name: str, arr: Any) -> None:
    out = out_dir / f"{name}.npy"
    np.save(out, arr)


def save_csv(out_dir: Path, name: str, rows: list[list[Any]], header: list[str] | None = None) -> None:
    out = out_dir / f"{name}.csv"
    with out.open("w", encoding="utf-8") as f:
        if header:
            f.write(",".join(map(str, header)) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")


def save_json(out_dir: Path, name: str, obj: Dict[str, Any]) -> None:
    out = out_dir / f"{name}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_plot_series(x: np.ndarray, series: dict, title: str, xlabel: str, ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        for label, y in series.items():
            plt.plot(x, y, label=label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
    except Exception:
        pass


