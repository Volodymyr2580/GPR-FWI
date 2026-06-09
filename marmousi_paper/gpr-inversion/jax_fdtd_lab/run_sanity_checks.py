"""Command-line sanity checks for local JAX gradient and L1-backward work."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from .cases import build_tiny_case
from .gradient_check import (
    run_epsilon_directional_gradient_check,
    run_l1_residual_backward_check,
    run_network_sigma_directional_gradient_check,
    run_sigma_directional_gradient_check,
)
from .jax_forward import describe_jax_environment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run local JAX FDTD gradient and L1 backward sanity checks."
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--finite-difference-step-sigma", type=float, default=2e-4)
    parser.add_argument("--finite-difference-step-network", type=float, default=1e-3)
    args = parser.parse_args(argv)

    case = build_tiny_case(steps=args.steps, dtype=np.float32)
    payload = {
        "jax_environment": asdict(describe_jax_environment()),
        "case": {
            "epsilon_shape": list(case.epsilon.shape),
            "n_sources": len(case.sources),
            "n_receivers": len(case.receivers),
            "steps": case.steps,
        },
        "checks": {
            "l1_backward_zero_subgradient": asdict(
                run_l1_residual_backward_check(zero_subgradient=True)
            ),
            "l1_backward_positive_zero_subgradient": asdict(
                run_l1_residual_backward_check(zero_subgradient=False)
            ),
            "epsilon_l2_directional": asdict(
                run_epsilon_directional_gradient_check(case, loss_norm="l2")
            ),
            "sigma_l2_directional": asdict(
                run_sigma_directional_gradient_check(
                    case,
                    finite_difference_step=args.finite_difference_step_sigma,
                    loss_norm="l2",
                )
            ),
            "sigma_l1_directional": asdict(
                run_sigma_directional_gradient_check(
                    case,
                    finite_difference_step=args.finite_difference_step_sigma,
                    loss_norm="l1",
                )
            ),
            "network_sigma_l2_directional": asdict(
                run_network_sigma_directional_gradient_check(
                    case,
                    finite_difference_step=args.finite_difference_step_network,
                    loss_norm="l2",
                )
            ),
            "network_sigma_l1_directional": asdict(
                run_network_sigma_directional_gradient_check(
                    case,
                    finite_difference_step=args.finite_difference_step_network,
                    loss_norm="l1",
                )
            ),
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
