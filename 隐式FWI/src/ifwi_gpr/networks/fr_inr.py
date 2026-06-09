"""Fourier reparameterized INR modules adapted from CVL-UESTC/FR-INR."""

from __future__ import annotations

import math

import torch
from torch import nn

from ifwi_gpr.parameterization import (
    ParameterBounds,
    ParameterStats,
    map_raw_to_parameters,
)
from ifwi_gpr.networks.siren import SineLayer


class FourierReparamLinear(nn.Module):
    """Linear layer whose weight is composed from fixed Fourier bases.

    FR-INR learns coefficients lambda and keeps the Fourier bases fixed:

        weight = lambda @ bases

    For our IFWI use case, this layer gives the coordinate network a stronger
    high-frequency representation without changing the physical parameter map.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        high_freq_num: int = 8,
        low_freq_num: int = 8,
        phi_num: int = 8,
        alpha: float = 0.01,
        omega0: float | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if high_freq_num < 0 or low_freq_num < 0:
            raise ValueError("frequency counts must be non-negative")
        if high_freq_num + low_freq_num <= 0:
            raise ValueError("at least one Fourier frequency is required")
        if phi_num <= 0:
            raise ValueError("phi_num must be positive")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.high_freq_num = int(high_freq_num)
        self.low_freq_num = int(low_freq_num)
        self.phi_num = int(phi_num)
        self.alpha = float(alpha)
        self.omega0 = None if omega0 is None else float(omega0)

        bases = self._build_bases()
        self.register_buffer("bases", bases)
        self.lamb = nn.Parameter(torch.empty(self.out_features, bases.shape[0]))
        self.bias = nn.Parameter(torch.zeros(self.out_features))
        self.reset_parameters()

    def _build_bases(self) -> torch.Tensor:
        phases = torch.arange(self.phi_num, dtype=torch.float32)
        phases = phases * (2.0 * math.pi / self.phi_num)

        low = torch.arange(1, self.low_freq_num + 1, dtype=torch.float32)
        if self.low_freq_num > 0:
            low = low / float(self.low_freq_num)
            period = 2.0 * math.pi / float(low[0])
        else:
            period = 2.0 * math.pi

        high = torch.arange(1, self.high_freq_num + 1, dtype=torch.float32)
        frequencies = torch.cat((low, high))
        points = torch.linspace(-period / 2.0, period / 2.0, self.in_features)

        rows = []
        for freq in frequencies:
            for phase in phases:
                rows.append(torch.cos(freq * points + phase))
        return self.alpha * torch.stack(rows, dim=0)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            basis_count = self.bases.shape[0]
            scale = math.sqrt(6.0 / basis_count)
            if self.omega0 is not None:
                scale /= self.omega0
            for idx in range(basis_count):
                denominator = torch.linalg.vector_norm(self.bases[idx]).clamp_min(1e-12)
                self.lamb[:, idx].uniform_(-scale / denominator, scale / denominator)
            self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.lamb @ self.bases
        return torch.nn.functional.linear(x, weight, self.bias)


class SineFourierReparamLayer(nn.Module):
    """Fourier reparameterized linear layer followed by scaled sine."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        high_freq_num: int = 8,
        low_freq_num: int = 8,
        phi_num: int = 8,
        alpha: float = 0.01,
        omega0: float = 30.0,
    ) -> None:
        super().__init__()
        self.omega0 = float(omega0)
        self.linear = FourierReparamLinear(
            in_features,
            out_features,
            high_freq_num=high_freq_num,
            low_freq_num=low_freq_num,
            phi_num=phi_num,
            alpha=alpha,
            omega0=omega0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * self.linear(x))


class FRINR(nn.Module):
    """Coordinate INR with optional Fourier reparameterized hidden layers."""

    def __init__(
        self,
        *,
        mode: str = "sin+fr",
        in_features: int = 2,
        out_features: int = 2,
        hidden_features: int = 128,
        hidden_layers: int = 4,
        high_freq_num: int = 8,
        low_freq_num: int = 8,
        phi_num: int = 8,
        alpha: float = 0.01,
        first_omega0: float = 30.0,
        hidden_omega0: float = 30.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if mode not in {"relu", "relu+fr", "sin", "sin+fr"}:
            raise ValueError("mode must be one of: relu, relu+fr, sin, sin+fr")

        layers: list[nn.Module] = []
        if mode.startswith("relu"):
            layers.extend([nn.Linear(in_features, hidden_features), nn.ReLU()])
        else:
            layers.append(
                SineLayer(
                    in_features,
                    hidden_features,
                    omega0=first_omega0,
                    is_first=True,
                )
            )
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        for _ in range(hidden_layers - 1):
            if mode == "relu":
                layers.extend([nn.Linear(hidden_features, hidden_features), nn.ReLU()])
            elif mode == "relu+fr":
                layers.extend(
                    [
                        FourierReparamLinear(
                            hidden_features,
                            hidden_features,
                            high_freq_num=high_freq_num,
                            low_freq_num=low_freq_num,
                            phi_num=phi_num,
                            alpha=alpha,
                        ),
                        nn.ReLU(),
                    ]
                )
            elif mode == "sin":
                layers.append(
                    SineLayer(
                        hidden_features,
                        hidden_features,
                        omega0=hidden_omega0,
                        is_first=False,
                    )
                )
            else:
                layers.append(
                    SineFourierReparamLayer(
                        hidden_features,
                        hidden_features,
                        high_freq_num=high_freq_num,
                        low_freq_num=low_freq_num,
                        phi_num=phi_num,
                        alpha=alpha,
                        omega0=hidden_omega0,
                    )
                )
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        self.net = nn.Sequential(*layers)
        self.final_linear = nn.Linear(hidden_features, out_features)
        self._reset_final(mode, hidden_features, hidden_omega0)

    def _reset_final(self, mode: str, hidden_features: int, hidden_omega0: float) -> None:
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_features)
            if mode.startswith("sin"):
                bound /= hidden_omega0
            self.final_linear.weight.uniform_(-bound, bound)
            self.final_linear.bias.zero_()

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.final_linear(self.net(coordinates))


class IFWIFrInrNetwork(nn.Module):
    """FR-INR coordinate network plus GPR physical parameter mapping."""

    def __init__(
        self,
        bounds: ParameterBounds | ParameterStats,
        *,
        shape: tuple[int, int],
        mode: str = "sin+fr",
        hidden_features: int = 128,
        hidden_layers: int = 4,
        high_freq_num: int = 8,
        low_freq_num: int = 8,
        phi_num: int = 8,
        alpha: float = 0.01,
        first_omega0: float = 30.0,
        hidden_omega0: float = 30.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.parameter_mapping = bounds
        self.bounds = bounds if isinstance(bounds, ParameterBounds) else None
        self.shape = tuple(shape)
        self.inr = FRINR(
            mode=mode,
            in_features=2,
            out_features=2,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            high_freq_num=high_freq_num,
            low_freq_num=low_freq_num,
            phi_num=phi_num,
            alpha=alpha,
            first_omega0=first_omega0,
            hidden_omega0=hidden_omega0,
            dropout=dropout,
        )

    def forward(self, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.inr(coordinates)
        epsilon, sigma = map_raw_to_parameters(raw, self.parameter_mapping)
        return epsilon.reshape(self.shape), sigma.reshape(self.shape)
