"""Correctness and performance probes for the JAX FDTD lab."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from dataclasses import replace
from typing import Any, Callable

import numpy as np

from .cases import TinyFdtdCase, build_smooth_case, build_tiny_case
from .cpu_baseline import (
    run_cpu_adjoint_gradient,
    run_cpu_baseline,
    run_cpu_forward_with_wavefield,
)
from .jax_adjoint import adjoint_gradient_jax_array, is_jax_available, run_jax_adjoint_gradient
from .jax_forward import (
    describe_jax_environment,
    forward_jax_array,
    run_jax_forward,
    run_jax_forward_with_wavefield,
)


@dataclass(frozen=True)
class ErrorSummary:
    max_abs: float
    relative_l2: float
    finite: bool


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run JAX FDTD correctness and performance probes.")
    parser.add_argument("--case", choices=("tiny", "smooth"), default="smooth")
    parser.add_argument("--xl", type=int, default=40)
    parser.add_argument("--zl", type=int, default=80)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--sources", type=int, default=4)
    parser.add_argument("--receivers", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--skip-wavefield", action="store_true")
    parser.add_argument("--skip-gradient", action="store_true")
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument(
        "--shot-batch-size",
        type=int,
        default=0,
        help="For JAX adjoint gradient, process this many shots at a time. 0 means all shots together.",
    )
    args = parser.parse_args(argv)

    case = _build_case(args)
    payload: dict[str, Any] = {
        "case": {
            "epsilon_shape": list(case.epsilon.shape),
            "n_sources": len(case.sources),
            "n_receivers": len(case.receivers),
            "steps": case.steps,
            "npml": case.npml,
        },
        "jax_environment": asdict(describe_jax_environment()),
        "gpu_memory_before_mb": _nvidia_memory_mb(),
    }

    cpu_forward_first_time, cpu_data = _time_call(lambda: run_cpu_baseline(case), block=False)
    cpu_forward_time, cpu_data = _time_repeated(
        lambda: run_cpu_baseline(case),
        args.repeats,
        block=False,
    )
    payload["cpu_forward_first_seconds"] = cpu_forward_first_time
    payload["cpu_forward_seconds"] = cpu_forward_time
    payload["cpu_forward_summary"] = _array_summary(cpu_data)

    if is_jax_available():
        if args.no_jit:
            jax_forward_fn = lambda: run_jax_forward(case)
            jax_compile_seconds = None
        else:
            jax_forward_fn = _make_jitted_forward(case)
            jax_compile_seconds, _ = _time_call(jax_forward_fn)
        jax_forward_time, jax_data = _time_repeated(jax_forward_fn, args.repeats)
        payload["jax_forward_compile_seconds"] = jax_compile_seconds
        payload["jax_forward_seconds"] = jax_forward_time
        payload["jax_forward_speedup_vs_cpu"] = cpu_forward_time / jax_forward_time if jax_forward_time > 0 else None
        payload["forward_error"] = asdict(_error_summary(jax_data, cpu_data))

        if args.skip_gradient and not args.skip_wavefield:
            cpu_wave_time, (cpu_wave_data, cpu_wavefield) = _time_call(
                lambda: run_cpu_forward_with_wavefield(case),
                block=False,
            )
            jax_wave_time, (jax_wave_data, jax_wavefield) = _time_call(
                lambda: run_jax_forward_with_wavefield(case)
            )
            payload["cpu_forward_with_wavefield_seconds"] = cpu_wave_time
            payload["jax_forward_with_wavefield_seconds"] = jax_wave_time
            payload["forward_wavefield_error"] = asdict(_error_summary(jax_wavefield, cpu_wavefield))
            payload["forward_wave_data_error"] = asdict(_error_summary(jax_wave_data, cpu_wave_data))

        elif not args.skip_gradient:
            cpu_wave_time, (cpu_wave_data, cpu_wavefield) = _time_call(
                lambda: run_cpu_forward_with_wavefield(case),
                block=False,
            )
            payload["cpu_forward_with_wavefield_seconds"] = cpu_wave_time
            if not args.skip_wavefield:
                jax_wave_time, (jax_wave_data, jax_wavefield) = _time_call(
                    lambda: run_jax_forward_with_wavefield(case)
                )
                payload["jax_forward_with_wavefield_seconds"] = jax_wave_time
                payload["forward_wavefield_error"] = asdict(_error_summary(jax_wavefield, cpu_wavefield))
                payload["forward_wave_data_error"] = asdict(_error_summary(jax_wave_data, cpu_wave_data))
            residual = (cpu_wave_data * args.residual_scale).astype(np.float32)
            cpu_grad_time, (cpu_grad_eps, cpu_grad_sig) = _time_call(
                lambda: run_cpu_adjoint_gradient(
                    case,
                    residual,
                    wavefield_data=cpu_wavefield,
                    sigma_required_gradient=True,
                ),
                block=False,
            )
            if args.no_jit:
                jax_grad_fn = lambda: run_jax_adjoint_gradient(case, residual, sigma_required_gradient=True)
                jax_grad_compile_seconds = None
            else:
                jax_grad_fn = _make_jitted_adjoint(
                    case,
                    residual,
                    shot_batch_size=args.shot_batch_size,
                )
                jax_grad_compile_seconds, _ = _time_call(jax_grad_fn)
            jax_grad_time, (jax_grad_eps, jax_grad_sig) = _time_repeated(jax_grad_fn, args.repeats)
            payload["cpu_adjoint_gradient_seconds"] = cpu_grad_time
            payload["jax_adjoint_total_compile_seconds"] = jax_grad_compile_seconds
            payload["jax_adjoint_total_seconds"] = jax_grad_time
            payload["jax_adjoint_shot_batch_size"] = args.shot_batch_size or len(case.sources)
            payload["jax_adjoint_total_speedup_vs_cpu_gradient_only"] = (
                cpu_grad_time / jax_grad_time if jax_grad_time > 0 else None
            )
            payload["epsilon_gradient_error"] = asdict(_error_summary(jax_grad_eps, cpu_grad_eps))
            payload["sigma_gradient_error"] = asdict(_error_summary(jax_grad_sig, cpu_grad_sig))

    payload["gpu_memory_after_mb"] = _nvidia_memory_mb()
    payload["jax_memory_stats"] = _jax_memory_stats()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _build_case(args) -> TinyFdtdCase:
    if args.case == "tiny":
        return build_tiny_case(steps=args.steps, dtype=np.float32)
    return build_smooth_case(
        xl=args.xl,
        zl=args.zl,
        steps=args.steps,
        n_sources=args.sources,
        n_receivers=args.receivers,
        dtype=np.float32,
    )


def _make_jitted_forward(case: TinyFdtdCase) -> Callable[[], np.ndarray]:
    import jax
    import jax.numpy as jnp

    epsilon = jnp.asarray(case.epsilon, dtype=jnp.float32)
    sigma = jnp.asarray(case.sigma, dtype=jnp.float32)
    fn = jax.jit(lambda eps, sig: forward_jax_array(case, eps, sig))

    def run() -> np.ndarray:
        return np.asarray(jax.device_get(fn(epsilon, sigma)))

    return run


def _make_jitted_adjoint(
    case: TinyFdtdCase,
    residual: np.ndarray,
    shot_batch_size: int = 0,
) -> Callable[[], tuple[np.ndarray, np.ndarray]]:
    import jax
    import jax.numpy as jnp

    epsilon = jnp.asarray(case.epsilon, dtype=jnp.float32)
    sigma = jnp.asarray(case.sigma, dtype=jnp.float32)

    if shot_batch_size <= 0 or shot_batch_size >= len(case.sources):
        residual_jax = jnp.asarray(residual, dtype=jnp.float32)
        fn = jax.jit(
            lambda eps, sig, res: adjoint_gradient_jax_array(
                case,
                eps,
                sig,
                res,
                sigma_required_gradient=True,
            )
        )

        def run() -> tuple[np.ndarray, np.ndarray]:
            grad_eps, grad_sig = fn(epsilon, sigma, residual_jax)
            return np.asarray(jax.device_get(grad_eps)), np.asarray(jax.device_get(grad_sig))

        return run

    batch_entries = []
    for start in range(0, len(case.sources), shot_batch_size):
        end = min(start + shot_batch_size, len(case.sources))
        batch_case = replace(case, sources=case.sources[start:end])
        batch_residual = jnp.asarray(residual[start:end], dtype=jnp.float32)
        batch_fn = jax.jit(
            lambda eps, sig, res, batch_case=batch_case: adjoint_gradient_jax_array(
                batch_case,
                eps,
                sig,
                res,
                sigma_required_gradient=True,
            )
        )
        batch_entries.append((batch_fn, batch_residual))

    def run_batched() -> tuple[np.ndarray, np.ndarray]:
        grad_eps_total = jnp.zeros_like(epsilon)
        grad_sig_total = jnp.zeros_like(sigma)
        for batch_fn, batch_residual in batch_entries:
            grad_eps, grad_sig = batch_fn(epsilon, sigma, batch_residual)
            grad_eps_total = grad_eps_total + grad_eps
            grad_sig_total = grad_sig_total + grad_sig
        return np.asarray(jax.device_get(grad_eps_total)), np.asarray(jax.device_get(grad_sig_total))

    return run_batched


def _time_call(fn: Callable[[], Any], block: bool = True) -> tuple[float, Any]:
    start = time.perf_counter()
    result = fn()
    if block:
        _block_until_ready(result)
    elapsed = time.perf_counter() - start
    return elapsed, result


def _time_repeated(fn: Callable[[], Any], repeats: int, block: bool = True) -> tuple[float, Any]:
    best = float("inf")
    best_result = None
    for _ in range(max(1, repeats)):
        elapsed, result = _time_call(fn, block=block)
        if elapsed < best:
            best = elapsed
            best_result = result
    return best, best_result


def _block_until_ready(value: Any) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    elif isinstance(value, (tuple, list)):
        for item in value:
            _block_until_ready(item)


def _error_summary(candidate: np.ndarray, reference: np.ndarray) -> ErrorSummary:
    candidate_arr = np.asarray(candidate)
    reference_arr = np.asarray(reference)
    diff = candidate_arr - reference_arr
    return ErrorSummary(
        max_abs=float(np.max(np.abs(diff))),
        relative_l2=float(
            np.linalg.norm(diff.ravel()) / max(np.linalg.norm(reference_arr.ravel()), 1e-30)
        ),
        finite=bool(np.isfinite(candidate_arr).all()),
    )


def _array_summary(value: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(value)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "finite": bool(np.isfinite(arr).all()),
    }


def _nvidia_memory_mb() -> dict[str, int] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None

    first_line = result.stdout.strip().splitlines()[0]
    used, total = [int(part.strip()) for part in first_line.split(",")[:2]]
    return {"used": used, "total": total}


def _jax_memory_stats() -> dict[str, Any] | None:
    if not is_jax_available():
        return None
    try:
        import jax

        stats = jax.devices()[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    return {str(key): value for key, value in stats.items()}


if __name__ == "__main__":
    main()
