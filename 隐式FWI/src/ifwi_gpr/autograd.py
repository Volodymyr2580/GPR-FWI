"""PyTorch autograd bridge for non-PyTorch GPR solvers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.autograd import Function

from ifwi_gpr.solver_bridge.cpu_mpi import CpuMpiSolverBridge, SolverSettings


class CpuMpiFDTDFunction(Function):
    """Wrap the verified CPU/MPI FDTD solver as a PyTorch autograd op."""

    @staticmethod
    def forward(
        ctx,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
        bridge: CpuMpiSolverBridge,
        source_list: list[tuple[int, int]],
        receiver_list: list[tuple[int, int]],
        settings: SolverSettings,
    ) -> torch.Tensor:
        epsilon_np = epsilon.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()

        result = bridge.forward(
            epsilon_np,
            sigma_np,
            source_list,
            receiver_list,
            settings,
            save_wavefield=True,
        )
        data_np, wavefield_data = result

        ctx.save_for_backward(epsilon, sigma)
        ctx.bridge = bridge
        ctx.source_list = source_list
        ctx.receiver_list = receiver_list
        ctx.settings = settings
        ctx.wavefield_data = wavefield_data

        return torch.as_tensor(data_np, dtype=epsilon.dtype, device=epsilon.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[Any, ...]:
        epsilon, sigma = ctx.saved_tensors
        grad_output_np = grad_output.detach().cpu().numpy()
        epsilon_np = epsilon.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()

        grad_eps, grad_sig = ctx.bridge.gradient(
            epsilon_np,
            sigma_np,
            grad_output_np,
            ctx.source_list,
            ctx.receiver_list,
            ctx.settings,
            wavefield_data=ctx.wavefield_data,
            sigma_required_gradient=True,
        )
        grad_eps, grad_sig = normalize_parameter_gradients(
            grad_eps,
            grad_sig,
            mode=ctx.settings.gradient_normalization,
            epsilon_weight=ctx.settings.epsilon_gradient_weight,
            sigma_weight=ctx.settings.sigma_gradient_weight,
        )

        grad_eps_tensor = torch.as_tensor(
            np.asarray(grad_eps),
            dtype=epsilon.dtype,
            device=epsilon.device,
        )
        grad_sig_tensor = torch.as_tensor(
            np.asarray(grad_sig),
            dtype=sigma.dtype,
            device=sigma.device,
        )
        return grad_eps_tensor, grad_sig_tensor, None, None, None, None


def cpu_mpi_forward(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    bridge: CpuMpiSolverBridge,
    source_list: list[tuple[int, int]],
    receiver_list: list[tuple[int, int]],
    settings: SolverSettings,
) -> torch.Tensor:
    """Call the CPU/MPI FDTD solver with PyTorch autograd support."""
    return CpuMpiFDTDFunction.apply(
        epsilon,
        sigma,
        bridge,
        source_list,
        receiver_list,
        settings,
    )


def normalize_parameter_gradients(
    grad_eps: np.ndarray,
    grad_sig: np.ndarray | None,
    *,
    mode: str = "none",
    epsilon_weight: float = 1.0,
    sigma_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Balance epsilon/sigma gradient channels before entering PyTorch."""
    normalized_mode = str(mode).lower()
    grad_eps_array = np.asarray(grad_eps)
    grad_sig_array = None if grad_sig is None else np.asarray(grad_sig)

    if normalized_mode in {"none", "off", "false", "0"}:
        return (
            grad_eps_array * float(epsilon_weight),
            None if grad_sig_array is None else grad_sig_array * float(sigma_weight),
        )
    if grad_sig_array is None:
        return grad_eps_array * float(epsilon_weight), None
    if normalized_mode not in {"match_rms", "rms", "per_parameter_rms"}:
        raise ValueError(f"Unsupported gradient_normalization mode: {mode}")

    eps_rms = _finite_rms(grad_eps_array)
    sig_rms = _finite_rms(grad_sig_array)
    if eps_rms <= 0.0 or sig_rms <= 0.0:
        return (
            grad_eps_array * float(epsilon_weight),
            grad_sig_array * float(sigma_weight),
        )

    target_rms = float(np.sqrt(eps_rms * sig_rms))
    grad_eps_balanced = grad_eps_array * (target_rms / eps_rms) * float(epsilon_weight)
    grad_sig_balanced = grad_sig_array * (target_rms / sig_rms) * float(sigma_weight)
    return grad_eps_balanced, grad_sig_balanced


def _finite_rms(values: np.ndarray) -> float:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(finite_values * finite_values)))
