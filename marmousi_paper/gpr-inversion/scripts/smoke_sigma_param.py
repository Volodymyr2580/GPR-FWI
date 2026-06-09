#!/usr/bin/env python3
"""Smoke test for sigma parameterization and beta-scaled gradients."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpr_inversion.experiments.overthrust.mode2_eps_then_sig import (  # noqa: E402
    twopara_epsfirst as runner,
)


def fake_forward(epsilon, sigma, source_list, receiver_list, dt, dx, dz, npml, freq, steps):
    return (
        np.ones((len(source_list), len(receiver_list), 4), dtype=np.float32)
        * float(np.mean(sigma))
    )


def fake_adjoint(
    epsilon,
    sigma,
    residual,
    source_list,
    receiver_list,
    dt,
    dx,
    dz,
    npml,
    freq,
    steps,
    sigma_required_gradient,
):
    grad_eps = np.ones_like(epsilon, dtype=np.float32) * 0.5
    grad_sig = (
        np.ones_like(sigma, dtype=np.float32) * 2.0
        if sigma_required_gradient
        else None
    )
    return grad_eps, grad_sig


def run_case(beta: float) -> tuple[float, float, float]:
    torch.manual_seed(2025)
    epsilon = (torch.ones((6, 8), dtype=torch.float32) * 4.0).requires_grad_()
    raw = (torch.randn((6, 8), dtype=torch.float32) * 0.05).requires_grad_()
    raw.retain_grad()

    sigmoid_sigma = torch.sigmoid(raw)
    log_min = torch.log(torch.tensor(3e-3))
    log_max = torch.log(torch.tensor(8.5e-3))
    sigma = torch.exp(sigmoid_sigma * (log_max - log_min) + log_min)
    sigma.retain_grad()

    out = runner.ForwardModelFunction.apply(
        epsilon,
        sigma,
        [(0, 0), (0, 5)],
        [(0, 0), (0, 2), (0, 4)],
        4e-11,
        0.02,
        0.02,
        2,
        4e8,
        4,
        1.0,
        1.0,
        0.0,
        0.03,
        beta,
        "log",
        True,
        False,
        "jax",
    )
    loss = out.square().mean()
    loss.backward()

    if not torch.isfinite(sigma.grad).all():
        raise RuntimeError("sigma gradient contains non-finite values")
    if not torch.isfinite(raw.grad).all():
        raise RuntimeError("raw log-sigma gradient contains non-finite values")
    if not torch.isfinite(epsilon.grad).all():
        raise RuntimeError("epsilon gradient contains non-finite values")

    return float(loss), float(sigma.grad.norm()), float(raw.grad.norm())


def main() -> None:
    runner._jax_forward_numpy = fake_forward
    runner._jax_adjoint_gradient_numpy = fake_adjoint

    loss_1, sigma_grad_1, raw_grad_1 = run_case(1.0)
    loss_025, sigma_grad_025, raw_grad_025 = run_case(0.25)

    print(f"loss_beta1={loss_1:.8e}")
    print(f"loss_beta025={loss_025:.8e}")
    print(f"sigma_grad_norm_beta1={sigma_grad_1:.8e}")
    print(f"sigma_grad_norm_beta025={sigma_grad_025:.8e}")
    print(f"sigma_grad_ratio={sigma_grad_025 / sigma_grad_1:.6f}")
    print(f"raw_grad_norm_beta1={raw_grad_1:.8e}")
    print(f"raw_grad_norm_beta025={raw_grad_025:.8e}")
    print(f"raw_grad_ratio={raw_grad_025 / raw_grad_1:.6f}")


if __name__ == "__main__":
    main()
