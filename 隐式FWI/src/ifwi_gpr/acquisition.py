"""Acquisition geometry builders for GPR IFWI experiments."""

from __future__ import annotations

from typing import Any

import numpy as np


GridPoint = tuple[int, int]


def build_acquisition(
    acquisition_cfg: dict[str, Any],
    shape: tuple[int, int],
) -> tuple[list[GridPoint], list[GridPoint]]:
    """Build source and receiver coordinates from config."""
    kind = str(acquisition_cfg.get("kind", "explicit")).lower()
    if kind == "explicit":
        return _as_grid_points(acquisition_cfg["sources"]), _as_grid_points(
            acquisition_cfg["receivers"]
        )
    if kind == "perimeter":
        offset = int(acquisition_cfg.get("offset", 0))
        source_offset = int(acquisition_cfg.get("source_offset", offset))
        receiver_offset = int(acquisition_cfg.get("receiver_offset", offset))
        sources = make_perimeter_points(
            shape,
            int(acquisition_cfg["source_count"]),
            offset=source_offset,
            start_index=int(acquisition_cfg.get("source_start_index", 0)),
        )
        receivers = make_perimeter_points(
            shape,
            int(acquisition_cfg["receiver_count"]),
            offset=receiver_offset,
            start_index=int(acquisition_cfg.get("receiver_start_index", 0)),
        )
        return sources, receivers
    raise ValueError(f"Unsupported acquisition kind: {kind}")


def make_perimeter_points(
    shape: tuple[int, int],
    count: int,
    *,
    offset: int = 0,
    start_index: int = 0,
) -> list[GridPoint]:
    """Return evenly spaced grid points along a rectangular perimeter.

    The point order starts on the top boundary, then walks clockwise. This is a
    practical paper-aligned approximation for the Cross-shape "surrounded"
    acquisition layout.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    perimeter = _perimeter_path(shape, offset=offset)
    if count > len(perimeter):
        raise ValueError(
            f"count={count} exceeds available perimeter points={len(perimeter)}"
        )
    indices = np.linspace(0, len(perimeter), count, endpoint=False, dtype=int)
    start = int(start_index) % len(perimeter)
    return [perimeter[(start + int(idx)) % len(perimeter)] for idx in indices]


def _perimeter_path(shape: tuple[int, int], *, offset: int = 0) -> list[GridPoint]:
    nx, nz = int(shape[0]), int(shape[1])
    if nx <= 1 or nz <= 1:
        raise ValueError("shape must have at least two points in each dimension")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    x0, z0 = offset, offset
    x1, z1 = nx - 1 - offset, nz - 1 - offset
    if x0 >= x1 or z0 >= z1:
        raise ValueError("offset is too large for the model shape")

    path: list[GridPoint] = []
    path.extend((x0, z) for z in range(z0, z1 + 1))
    path.extend((x, z1) for x in range(x0 + 1, x1 + 1))
    path.extend((x1, z) for z in range(z1 - 1, z0 - 1, -1))
    path.extend((x, z0) for x in range(x1 - 1, x0, -1))
    return path


def _as_grid_points(values: list[Any]) -> list[GridPoint]:
    return [(int(item[0]), int(item[1])) for item in values]
