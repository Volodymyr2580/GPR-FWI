"""Synthetic model builders for smoke tests only, not paper experiments."""

from __future__ import annotations

import numpy as np


def build_cross_shape_model(
    shape: tuple[int, int] = (101, 101),
    *,
    epsilon_background: float = 4.0,
    sigma_background: float = 0.003,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the two-anomaly Cross-shape model used for lightweight tests.

    Conductivity is returned in S/m. The paper often reports mS/m, so 3 mS/m
    is represented here as 0.003 S/m.
    """
    nx, nz = shape
    epsilon = np.full((nx, nz), epsilon_background, dtype=np.float32)
    sigma = np.full((nx, nz), sigma_background, dtype=np.float32)

    _draw_cross(epsilon, center=(30, 30), half_length=12, thickness=3, value=1.0)
    _draw_cross(sigma, center=(30, 30), half_length=12, thickness=3, value=0.0001)
    _draw_cross(epsilon, center=(70, 70), half_length=12, thickness=3, value=8.0)
    _draw_cross(sigma, center=(70, 70), half_length=12, thickness=3, value=0.01)
    return epsilon, sigma


def _draw_cross(
    array: np.ndarray,
    *,
    center: tuple[int, int],
    half_length: int,
    thickness: int,
    value: float,
) -> None:
    cx, cz = center
    hx = int(half_length)
    ht = int(thickness)
    nx, nz = array.shape
    x0, x1 = max(0, cx - hx), min(nx, cx + hx + 1)
    z0, z1 = max(0, cz - hx), min(nz, cz + hx + 1)
    tx0, tx1 = max(0, cx - ht), min(nx, cx + ht + 1)
    tz0, tz1 = max(0, cz - ht), min(nz, cz + ht + 1)
    array[tx0:tx1, z0:z1] = value
    array[x0:x1, tz0:tz1] = value
