"""Physical parameter transforms for IFWI networks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ParameterBounds:
    """Physical bounds used to map raw network output to GPR parameters."""

    epsilon_min: float
    epsilon_max: float
    sigma_min: float
    sigma_max: float

    def validate(self) -> None:
        if self.epsilon_min >= self.epsilon_max:
            raise ValueError("epsilon_min must be smaller than epsilon_max")
        if self.sigma_min >= self.sigma_max:
            raise ValueError("sigma_min must be smaller than sigma_max")
        if self.sigma_min < 0.0:
            raise ValueError("sigma_min must be non-negative")


@dataclass(frozen=True)
class ParameterStats:
    """Mean/std mapping used by the IFWI paper.

    Conductivity is stored in S/m in this codebase, even though the paper
    often reports it in mS/m.
    """

    epsilon_mean: float
    epsilon_std: float
    sigma_mean: float
    sigma_std: float
    epsilon_min: float | None = None
    epsilon_max: float | None = None
    sigma_min: float | None = None
    sigma_max: float | None = None

    def validate(self) -> None:
        if self.epsilon_std <= 0.0:
            raise ValueError("epsilon_std must be positive")
        if self.sigma_std <= 0.0:
            raise ValueError("sigma_std must be positive")
        if self.epsilon_min is not None and self.epsilon_max is not None:
            if self.epsilon_min >= self.epsilon_max:
                raise ValueError("epsilon_min must be smaller than epsilon_max")
        if self.sigma_min is not None and self.sigma_max is not None:
            if self.sigma_min >= self.sigma_max:
                raise ValueError("sigma_min must be smaller than sigma_max")
        if self.sigma_min is not None and self.sigma_min < 0.0:
            raise ValueError("sigma_min must be non-negative")


def map_raw_to_physical(
    raw: torch.Tensor,
    bounds: ParameterBounds,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map unconstrained network output to epsilon_r and sigma in S/m."""
    bounds.validate()
    if raw.shape[-1] != 2:
        raise ValueError("raw output must have two channels: epsilon and sigma")

    scaled = torch.sigmoid(raw)
    epsilon = bounds.epsilon_min + scaled[..., 0] * (
        bounds.epsilon_max - bounds.epsilon_min
    )
    sigma = bounds.sigma_min + scaled[..., 1] * (bounds.sigma_max - bounds.sigma_min)
    return epsilon, sigma


def map_raw_standardized(
    raw: torch.Tensor,
    stats: ParameterStats,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map network output m_tilde to physical parameters via mean/std."""
    stats.validate()
    if raw.shape[-1] != 2:
        raise ValueError("raw output must have two channels: epsilon and sigma")
    epsilon = raw[..., 0] * stats.epsilon_std + stats.epsilon_mean
    sigma = raw[..., 1] * stats.sigma_std + stats.sigma_mean
    if stats.epsilon_min is not None or stats.epsilon_max is not None:
        epsilon = torch.clamp(
            epsilon,
            min=stats.epsilon_min,
            max=stats.epsilon_max,
        )
    if stats.sigma_min is not None or stats.sigma_max is not None:
        sigma = torch.clamp(
            sigma,
            min=stats.sigma_min,
            max=stats.sigma_max,
        )
    return epsilon, sigma


def map_raw_to_parameters(
    raw: torch.Tensor,
    mapping: ParameterBounds | ParameterStats,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map raw network output according to the configured parameterization."""
    if isinstance(mapping, ParameterBounds):
        return map_raw_to_physical(raw, mapping)
    if isinstance(mapping, ParameterStats):
        return map_raw_standardized(raw, mapping)
    raise TypeError(f"Unsupported parameter mapping: {type(mapping)!r}")
