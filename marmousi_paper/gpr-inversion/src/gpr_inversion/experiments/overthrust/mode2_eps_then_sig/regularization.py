"""Experiment-specific regularization bridge for the mode2 eps-then-sig runner."""

import torch
from torch.autograd import Function

from gpr_inversion.regularization import (
    combine_data_and_tv_gradient,
    compute_tv_gradient,
    normalize_torch_gradient,
)
from .gradient import compute_laplacian_2d, compute_tikhonov_gradient


class TikhonovRegularizationFunction(Function):
    """Torch autograd bridge for the existing NumPy Tikhonov gradient code."""

    @staticmethod
    def forward(ctx, epsilon, sigma, alpha_tikhonov_eps, alpha_tikhonov_sig, dx, dz):
        epsilon_np = epsilon.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()

        laplacian_eps_np = compute_laplacian_2d(epsilon_np, dx, dz)
        laplacian_sig_np = compute_laplacian_2d(sigma_np, dx, dz)

        laplacian_eps = torch.from_numpy(laplacian_eps_np).to(epsilon.device)
        laplacian_sig = torch.from_numpy(laplacian_sig_np).to(sigma.device)

        reg_loss_eps = 1e-10 * alpha_tikhonov_eps * torch.sum(laplacian_eps**2)
        reg_loss_sig = 1e-10 * alpha_tikhonov_sig * torch.sum(laplacian_sig**2)
        reg_loss = reg_loss_eps + reg_loss_sig

        ctx.save_for_backward(epsilon, sigma, laplacian_eps, laplacian_sig)
        ctx.alpha_tikhonov_eps = alpha_tikhonov_eps
        ctx.alpha_tikhonov_sig = alpha_tikhonov_sig
        ctx.dx = dx
        ctx.dz = dz

        return reg_loss

    @staticmethod
    def backward(ctx, grad_output):
        epsilon, sigma, _laplacian_eps, _laplacian_sig = ctx.saved_tensors
        alpha_tikhonov_eps = ctx.alpha_tikhonov_eps
        alpha_tikhonov_sig = ctx.alpha_tikhonov_sig
        dx = ctx.dx
        dz = ctx.dz

        epsilon_np = epsilon.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()

        tikhonov_grad_eps = compute_tikhonov_gradient(
            epsilon_np, dx, dz, alpha_tikhonov_eps
        )
        tikhonov_grad_sig = compute_tikhonov_gradient(
            sigma_np, dx, dz, alpha_tikhonov_sig
        )

        grad_eps = torch.from_numpy(tikhonov_grad_eps).to(epsilon.device) * grad_output
        grad_sig = torch.from_numpy(tikhonov_grad_sig).to(sigma.device) * grad_output

        return grad_eps, grad_sig, None, None, None, None


def tikhonov_reg(epsilon, sigma, alpha_tikhonov_eps, alpha_tikhonov_sig, dx=0.02, dz=0.02):
    """Compute the Tikhonov regularization loss through the custom autograd bridge."""
    return TikhonovRegularizationFunction.apply(
        epsilon, sigma, alpha_tikhonov_eps, alpha_tikhonov_sig, dx, dz
    )
