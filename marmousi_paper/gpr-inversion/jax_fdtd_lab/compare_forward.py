"""Command-line smoke comparison for CPU baseline and optional JAX forward."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np

from .cases import build_tiny_case
from .cpu_baseline import run_cpu_baseline, summarize_array
from .jax_forward import describe_jax_environment, run_jax_forward


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare tiny CPU and JAX FDTD forward outputs.")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument(
        "--backend",
        choices=("cpu", "jax", "both"),
        default="both",
        help="Which backend to run. JAX requires jax to be installed in the active environment.",
    )
    args = parser.parse_args(argv)

    case = build_tiny_case(steps=args.steps)
    payload: dict[str, object] = {
        "case": {
            "epsilon_shape": list(case.epsilon.shape),
            "n_sources": len(case.sources),
            "n_receivers": len(case.receivers),
            "steps": case.steps,
            "npml": case.npml,
        },
        "jax_environment": asdict(describe_jax_environment()),
    }

    cpu_data = None
    if args.backend in {"cpu", "both"}:
        cpu_data = run_cpu_baseline(case)
        payload["cpu"] = asdict(summarize_array(cpu_data))

    if args.backend in {"jax", "both"}:
        try:
            jax_data = run_jax_forward(case)
        except RuntimeError as exc:
            payload["jax"] = {"skipped": True, "reason": str(exc)}
        else:
            payload["jax"] = asdict(summarize_array(jax_data))
            if cpu_data is not None:
                diff = np.asarray(jax_data) - np.asarray(cpu_data)
                payload["comparison"] = {
                    "max_abs_error": float(np.max(np.abs(diff))),
                    "relative_l2_error": float(
                        np.linalg.norm(diff.ravel())
                        / max(np.linalg.norm(np.asarray(cpu_data).ravel()), 1e-30)
                    ),
                }

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

