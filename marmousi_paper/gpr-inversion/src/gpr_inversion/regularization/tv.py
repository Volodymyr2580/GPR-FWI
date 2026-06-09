"""Total-variation regularization helpers."""

from __future__ import annotations

import numpy as np
import torch


def compute_tv_gradient(model, dx: float, dz: float) -> np.ndarray:
    """Compute isotropic TV gradient for a 2D [z, x] model."""
    eps = 1e-8
    model_np = np.asarray(model, dtype=np.float32)
    padded = np.pad(model_np, ((1, 1), (1, 1)), mode="edge")

    grad_z = (padded[2:, 1:-1] - padded[1:-1, 1:-1]) / dz
    grad_x = (padded[1:-1, 2:] - padded[1:-1, 1:-1]) / dx

    norm = np.sqrt(grad_x**2 + grad_z**2 + eps)
    flux_z = grad_z / norm
    flux_x = grad_x / norm

    flux_z_pad = np.pad(flux_z, ((1, 0), (0, 0)), mode="constant")
    flux_x_pad = np.pad(flux_x, ((0, 0), (1, 0)), mode="constant")

    div_z = (flux_z_pad[1:, :] - flux_z_pad[:-1, :]) / dz
    div_x = (flux_x_pad[:, 1:] - flux_x_pad[:, :-1]) / dx
    return -(div_z + div_x)


def normalize_torch_gradient(gradient):
    """Normalize a torch gradient by its maximum absolute value."""
    max_abs = torch.max(torch.abs(gradient))
    if max_abs > 0:
        return gradient / max_abs
    return gradient


def combine_data_and_tv_gradient(
    data_gradient,
    model_np,
    dx: float,
    dz: float,
    alpha_data: float,
    alpha_tv: float,
    device,
):
    """Combine normalized data and TV gradients."""
    tv_gradient = compute_tv_gradient(model_np, dx, dz)
    tv_gradient_tensor = torch.from_numpy(tv_gradient).to(device).float()
    combined = normalize_torch_gradient(data_gradient) + alpha_tv * normalize_torch_gradient(tv_gradient_tensor)
    return normalize_torch_gradient(combined) * alpha_data
