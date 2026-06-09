"""CPU/Numba baseline runner for the JAX FDTD lab."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cases import TinyFdtdCase


@dataclass(frozen=True)
class ArraySummary:
    """Compact numeric summary for comparing forward-model outputs."""

    shape: tuple[int, ...]
    dtype: str
    minimum: float
    maximum: float
    mean: float
    finite: bool


def summarize_array(value: np.ndarray) -> ArraySummary:
    """Return a small summary that is stable enough for logs and progress notes."""
    arr = np.asarray(value)
    return ArraySummary(
        shape=tuple(arr.shape),
        dtype=str(arr.dtype),
        minimum=float(np.min(arr)),
        maximum=float(np.max(arr)),
        mean=float(np.mean(arr)),
        finite=bool(np.isfinite(arr).all()),
    )


def run_cpu_baseline(case: TinyFdtdCase) -> np.ndarray:
    """Run the existing production CPU/Numba FDTD as the correctness baseline."""
    from gpr_inversion.experiments.overthrust.mode2_eps_then_sig.forward import (
        forward_model,
    )

    return forward_model(
        case.epsilon,
        case.sigma,
        list(case.sources),
        list(case.receivers),
        case.dt,
        case.dx,
        case.dz,
        case.npml,
        case.freq,
        steps=case.steps,
        save_wavefield=False,
    )


def run_cpu_forward_with_wavefield(case: TinyFdtdCase) -> tuple[np.ndarray, np.ndarray]:
    """Run the original CPU/Numba forward model and return receiver data plus Ey wavefields."""
    from gpr_inversion.experiments.overthrust.mode2_eps_then_sig.forward import (
        forward_model,
    )

    data, wavefield = forward_model(
        case.epsilon,
        case.sigma,
        list(case.sources),
        list(case.receivers),
        case.dt,
        case.dx,
        case.dz,
        case.npml,
        case.freq,
        steps=case.steps,
        save_wavefield=True,
    )
    return np.asarray(data), np.asarray(wavefield)


def run_cpu_forward_with_illumination(case: TinyFdtdCase) -> tuple[np.ndarray, np.ndarray]:
    """Run the original CPU/Numba forward model and return receiver data plus illumination."""
    from gpr_inversion.experiments.overthrust.mode2_eps_then_sig.forward import (
        forward_model,
    )

    data, illumination = forward_model(
        case.epsilon,
        case.sigma,
        list(case.sources),
        list(case.receivers),
        case.dt,
        case.dx,
        case.dz,
        case.npml,
        case.freq,
        steps=case.steps,
        save_wavefield=False,
        return_illumination=True,
    )
    return np.asarray(data), np.asarray(illumination)


def run_cpu_adjoint_gradient(
    case: TinyFdtdCase,
    residual: np.ndarray,
    wavefield_data: np.ndarray | None = None,
    sigma_required_gradient: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run the original adjoint-gradient code used by `gpr-inversion`."""
    from gpr_inversion.experiments.overthrust.mode2_eps_then_sig.gradient import (
        compute_gradient,
    )

    if wavefield_data is None:
        _data, wavefield_data = run_cpu_forward_with_wavefield(case)

    return compute_gradient(
        case.epsilon,
        case.sigma,
        np.asarray(residual),
        list(case.sources),
        list(case.receivers),
        case.dt,
        case.dx,
        case.dz,
        case.npml,
        case.freq,
        case.steps,
        sigma_required_gradient=sigma_required_gradient,
        wavefield_data=wavefield_data,
    )
