"""Small deterministic FDTD cases for CPU/JAX comparison."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class TinyFdtdCase:
    """A deliberately small GPR FDTD case used as a correctness baseline."""

    epsilon: np.ndarray
    sigma: np.ndarray
    sources: tuple[tuple[int, int], ...]
    receivers: tuple[tuple[int, int], ...]
    dt: float
    dx: float
    dz: float
    npml: int
    freq: float
    steps: int


def build_tiny_case(steps: int = 24, dtype=np.float64) -> TinyFdtdCase:
    """Build a tiny heterogeneous model that runs quickly in CPU tests."""
    xl, zl = 20, 40
    x = np.linspace(0.0, 1.0, xl, dtype=dtype)[:, None]
    z = np.linspace(0.0, 1.0, zl, dtype=dtype)[None, :]
    epsilon = 4.0 + 0.4 * x + 0.2 * np.sin(2.0 * np.pi * z)
    epsilon = epsilon.astype(dtype, copy=False)
    sigma = (epsilon * 1e-3).astype(dtype, copy=False)

    return TinyFdtdCase(
        epsilon=epsilon,
        sigma=sigma,
        sources=((0, 10), (0, 20)),
        receivers=((0, 8), (0, 12), (0, 20), (0, 28)),
        dt=4e-11,
        dx=0.02,
        dz=0.02,
        npml=10,
        freq=4e8,
        steps=steps,
    )


def build_smooth_case(
    xl: int = 40,
    zl: int = 80,
    steps: int = 120,
    n_sources: int = 4,
    n_receivers: int = 16,
    dtype=np.float64,
) -> TinyFdtdCase:
    """Build a deterministic smooth heterogeneous model for scaled validation."""
    x = np.linspace(0.0, 1.0, xl, dtype=dtype)[:, None]
    z = np.linspace(0.0, 1.0, zl, dtype=dtype)[None, :]
    epsilon = (
        4.0
        + 0.4 * x
        + 0.2 * np.sin(2.0 * np.pi * z)
        + 0.15 * np.cos(np.pi * x + 3.0 * np.pi * z)
    )
    epsilon = epsilon.astype(dtype, copy=False)
    sigma = (epsilon * 1e-3).astype(dtype, copy=False)

    return TinyFdtdCase(
        epsilon=epsilon,
        sigma=sigma,
        sources=tuple((0, z_idx) for z_idx in _even_positions(zl, n_sources)),
        receivers=tuple((0, z_idx) for z_idx in _even_positions(zl, n_receivers)),
        dt=4e-11,
        dx=0.02,
        dz=0.02,
        npml=10,
        freq=4e8,
        steps=steps,
    )


def with_steps(case: TinyFdtdCase, steps: int) -> TinyFdtdCase:
    """Return a copy of a case with a different time-step count."""
    return replace(case, steps=steps)


def _even_positions(length: int, count: int) -> tuple[int, ...]:
    if count <= 0:
        return ()
    if count == 1:
        return (length // 2,)

    margin = max(1, min(10, length // 5))
    start = margin
    stop = max(start, length - 1 - margin)
    values = np.linspace(start, stop, count)
    positions = [int(round(value)) for value in values]
    return tuple(max(0, min(length - 1, value)) for value in positions)
