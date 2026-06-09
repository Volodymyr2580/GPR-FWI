"""Output helpers that avoid deleting previous runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def unique_run_dir(path: str | Path) -> Path:
    """Create a run directory, adding a suffix if it already exists."""
    base = Path(path)
    candidate = base
    idx = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_run{idx:03d}")
        idx += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_snapshot(run_dir: Path, epoch: int, epsilon: np.ndarray, sigma: np.ndarray) -> None:
    np.save(run_dir / f"epoch_{epoch}_epsilon.npy", epsilon)
    np.save(run_dir / f"epoch_{epoch}_sigma.npy", sigma)

