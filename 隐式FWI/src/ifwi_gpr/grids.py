"""Coordinate-grid helpers."""

from __future__ import annotations

import torch


def make_normalized_grid(
    shape: tuple[int, int],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return flattened (x, z) coordinates normalized to [-1, 1]."""
    nx, nz = shape
    if nx <= 0 or nz <= 0:
        raise ValueError("shape entries must be positive")
    x = torch.linspace(-1.0, 1.0, nx, device=device, dtype=dtype)
    z = torch.linspace(-1.0, 1.0, nz, device=device, dtype=dtype)
    xx, zz = torch.meshgrid(x, z, indexing="ij")
    return torch.stack((xx, zz), dim=-1).reshape(-1, 2)

