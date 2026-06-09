"""SIREN-style coordinate networks for IFWI."""

from __future__ import annotations

import math

import torch
from torch import nn

from ifwi_gpr.parameterization import (
    ParameterBounds,
    ParameterStats,
    map_raw_to_parameters,
)


class SineLayer(nn.Module):
    """Linear layer followed by the scaled sine activation used by SIREN."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        omega0: float = 30.0,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.omega0 = float(omega0)
        self.is_first = bool(is_first)
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * self.linear(coordinates))


class Siren(nn.Module):
    """Small SIREN MLP that maps normalized coordinates to raw parameters."""

    def __init__(
        self,
        *,
        in_features: int = 2,
        out_features: int = 2,
        hidden_features: int = 128,
        hidden_layers: int = 4,
        omega0: float = 30.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        layers: list[nn.Module] = [
            SineLayer(in_features, hidden_features, omega0=omega0, is_first=True)
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        for _ in range(hidden_layers - 1):
            layers.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    omega0=omega0,
                    is_first=False,
                )
            )
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        self.net = nn.Sequential(*layers)
        self.final_linear = nn.Linear(hidden_features, out_features)
        self.reset_final_layer(hidden_features)

    def reset_final_layer(self, hidden_features: int) -> None:
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_features)
            self.final_linear.weight.uniform_(-bound, bound)
            self.final_linear.bias.zero_()

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.final_linear(self.net(coordinates))


class IFWINetwork(nn.Module):
    """Coordinate network plus physical parameter mapping.

    Input coordinates are expected to be normalized to [-1, 1]. The output is a
    pair of 2-D tensors: relative permittivity and conductivity in S/m.
    """

    def __init__(
        self,
        bounds: ParameterBounds | ParameterStats,
        *,
        shape: tuple[int, int],
        hidden_features: int = 128,
        hidden_layers: int = 4,
        omega0: float = 30.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.parameter_mapping = bounds
        self.bounds = bounds if isinstance(bounds, ParameterBounds) else None
        self.shape = tuple(shape)
        self.siren = Siren(
            in_features=2,
            out_features=2,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            omega0=omega0,
            dropout=dropout,
        )

    def forward(self, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.siren(coordinates)
        epsilon, sigma = map_raw_to_parameters(raw, self.parameter_mapping)
        return epsilon.reshape(self.shape), sigma.reshape(self.shape)
