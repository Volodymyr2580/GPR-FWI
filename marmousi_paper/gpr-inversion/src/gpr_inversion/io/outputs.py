"""Safe output directory and model snapshot helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def ensure_unique_dir(target_dir: str | Path) -> Path:
    """Create a unique output directory without deleting previous results."""
    target = Path(target_dir)
    candidate = target
    if candidate.exists():
        run_idx = 1
        while candidate.exists():
            candidate = target.with_name(f"{target.name}_run{run_idx:03d}")
            run_idx += 1
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def save_model_snapshot(
    output_dir: str | Path,
    epoch: int,
    epsilon,
    sigma=None,
    save_bin: bool = False,
) -> list[Path]:
    """Save epsilon/sigma raw arrays for later plotting and paper processing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    eps_path = output_path / f"epoch_{epoch}_epsilon.npy"
    epsilon_array = _as_numpy(epsilon)
    np.save(eps_path, epsilon_array)
    saved.append(eps_path)
    if save_bin:
        eps_bin = output_path / f"epoch_{epoch}_epsilon.bin"
        epsilon_array.astype(np.float32).tofile(eps_bin)
        saved.append(eps_bin)

    if sigma is not None:
        sig_path = output_path / f"epoch_{epoch}_sigma.npy"
        sigma_array = _as_numpy(sigma)
        np.save(sig_path, sigma_array)
        saved.append(sig_path)
        if save_bin:
            sig_bin = output_path / f"epoch_{epoch}_sigma.bin"
            sigma_array.astype(np.float32).tofile(sig_bin)
            saved.append(sig_bin)

    return saved


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)
