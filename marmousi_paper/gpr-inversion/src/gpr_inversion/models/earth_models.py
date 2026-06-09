"""Earth-model loading and initialization utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_epsilon_model(model: str, data_path: str | Path) -> np.ndarray:
    """Load a Marmousi or OverThrust epsilon model from a NumPy file."""
    normalized = model.lower()
    if normalized not in {"marmousi", "overthrust"}:
        raise ValueError("model must be 'marmousi' or 'overthrust'")
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file does not exist: {path}")
    return np.load(path)


def resize_model(model: np.ndarray, target_shape: tuple[int, int], order: int = 1) -> np.ndarray:
    """Resize a 2D model to a target shape using scipy zoom."""
    from scipy.ndimage import zoom

    if model.ndim != 2:
        raise ValueError("model must be a 2D array")
    zoom_factors = (target_shape[0] / model.shape[0], target_shape[1] / model.shape[1])
    return zoom(model, zoom_factors, order=order)


def build_sigma_from_epsilon(epsilon: np.ndarray, scale: float = 1e-3) -> np.ndarray:
    """Create the current project convention for sigma from epsilon."""
    return np.asarray(epsilon, dtype=np.float32) * scale


def build_initial_model(
    true_model: np.ndarray,
    kind: str,
    smooth_sigma: float = 5.0,
    constant_value: float | None = None,
) -> np.ndarray:
    """Build an initial model used by traditional and network experiments."""
    normalized = kind.lower()
    if normalized == "smooth":
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(true_model, smooth_sigma)
    if normalized == "inverse_smooth":
        from scipy.ndimage import gaussian_filter

        return 1.0 / gaussian_filter(1.0 / true_model, smooth_sigma)
    if normalized == "uniform":
        value = float(np.mean(true_model) if constant_value is None else constant_value)
        return np.full_like(true_model, value)
    if normalized == "linear":
        top = float(np.mean(true_model[0, :]))
        bottom = float(np.mean(true_model[-1, :]))
        profile = np.linspace(top, bottom, true_model.shape[0], dtype=true_model.dtype)
        return np.repeat(profile[:, None], true_model.shape[1], axis=1)
    raise ValueError("kind must be one of: smooth, inverse_smooth, uniform, linear")
