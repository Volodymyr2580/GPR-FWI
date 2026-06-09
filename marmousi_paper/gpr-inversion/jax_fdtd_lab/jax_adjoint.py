"""JAX implementation of the original `gpr-inversion` adjoint-gradient path."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .cases import TinyFdtdCase
from .jax_forward import (
    _build_static_jax_params,
    _update_e,
    _update_h,
    _with_material_params,
    _jax_default_device_context,
    _clear_jax_caches,
    _is_resource_exhausted_error,
    _case_static_key,
    forward_jax_array,
    is_jax_available,
)


def run_jax_adjoint_gradient(
    case: TinyFdtdCase,
    residual: np.ndarray,
    sigma_required_gradient: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run a JAX port of the original adjoint gradient and return NumPy arrays."""
    if not is_jax_available():
        raise RuntimeError("JAX is not installed in this Python environment.")

    import jax
    import jax.numpy as jnp

    with _jax_default_device_context(jax):
        epsilon = jnp.asarray(case.epsilon, dtype=jnp.float32)
        sigma = jnp.asarray(case.sigma, dtype=jnp.float32)
        residual_jax = jnp.asarray(residual, dtype=jnp.float32)
        grad_eps, grad_sig = adjoint_gradient_jax_array(
            case,
            epsilon,
            sigma,
            residual_jax,
            sigma_required_gradient=sigma_required_gradient,
        )
    grad_eps_np = np.asarray(jax.device_get(grad_eps))
    if sigma_required_gradient:
        return grad_eps_np, np.asarray(jax.device_get(grad_sig))
    return grad_eps_np, None


_JITTED_ADJOINT_CACHE: dict[tuple[Any, ...], Any] = {}


def run_jax_adjoint_gradient_jitted(
    case: TinyFdtdCase,
    residual: np.ndarray,
    sigma_required_gradient: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run the manual adjoint gradient through a cached JIT kernel."""
    if not is_jax_available():
        raise RuntimeError("JAX is not installed in this Python environment.")

    import jax
    import jax.numpy as jnp

    key = ("adjoint", _case_static_key(case), bool(sigma_required_gradient))
    with _jax_default_device_context(jax):
        fn = _JITTED_ADJOINT_CACHE.get(key)
        if fn is None:
            if sigma_required_gradient:
                fn = jax.jit(
                    lambda eps, sig, res: adjoint_gradient_jax_array(
                        case,
                        eps,
                        sig,
                        res,
                        sigma_required_gradient=True,
                    )
                )
            else:
                fn = jax.jit(
                    lambda eps, sig, res: adjoint_gradient_jax_array(
                        case,
                        eps,
                        sig,
                        res,
                        sigma_required_gradient=False,
                    )[0]
                )
            _JITTED_ADJOINT_CACHE[key] = fn

        epsilon = jnp.asarray(case.epsilon, dtype=jnp.float32)
        sigma = jnp.asarray(case.sigma, dtype=jnp.float32)
        residual_jax = jnp.asarray(residual, dtype=jnp.float32)
        result = fn(epsilon, sigma, residual_jax)

    if sigma_required_gradient:
        grad_eps, grad_sig = result
        return np.asarray(jax.device_get(grad_eps)), np.asarray(jax.device_get(grad_sig))
    return np.asarray(jax.device_get(result)), None


def run_jax_adjoint_gradient_batched(
    case: TinyFdtdCase,
    residual: np.ndarray,
    sigma_required_gradient: bool = True,
    shot_batch_size: int = 5,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run the manual adjoint gradient in source batches to control memory."""
    n_shots = len(case.sources)
    if n_shots == 0:
        grad_eps = np.zeros_like(case.epsilon, dtype=np.float32)
        grad_sig = np.zeros_like(case.sigma, dtype=np.float32) if sigma_required_gradient else None
        return grad_eps, grad_sig
    if shot_batch_size <= 0 or shot_batch_size >= n_shots:
        return run_jax_adjoint_gradient_jitted(
            case,
            residual,
            sigma_required_gradient=sigma_required_gradient,
        )

    residual = np.asarray(residual, dtype=np.float32)
    grad_eps_total = np.zeros_like(case.epsilon, dtype=np.float32)
    grad_sig_total = np.zeros_like(case.sigma, dtype=np.float32) if sigma_required_gradient else None
    start = 0
    current_batch_size = min(shot_batch_size, n_shots)
    while start < n_shots:
        current_batch_size = min(current_batch_size, n_shots - start)
        while True:
            stop = start + current_batch_size
            batch_case = replace(case, sources=case.sources[start:stop])
            try:
                batch_grad_eps, batch_grad_sig = run_jax_adjoint_gradient_jitted(
                    batch_case,
                    residual[start:stop],
                    sigma_required_gradient=sigma_required_gradient,
                )
                break
            except Exception as exc:
                if current_batch_size <= 1 or not _is_resource_exhausted_error(exc):
                    raise
                _clear_jax_caches()
                current_batch_size = max(1, current_batch_size // 2)
                print(
                    "JAX adjoint OOM; retrying with "
                    f"shot_batch_size={current_batch_size}",
                    flush=True,
                )
        grad_eps_total += np.asarray(batch_grad_eps, dtype=np.float32)
        if sigma_required_gradient and grad_sig_total is not None and batch_grad_sig is not None:
            grad_sig_total += np.asarray(batch_grad_sig, dtype=np.float32)
        start += current_batch_size
    return grad_eps_total, grad_sig_total


def adjoint_gradient_jax_array(
    case: TinyFdtdCase,
    epsilon: Any,
    sigma: Any,
    residual: Any,
    sigma_required_gradient: bool = True,
):
    """Compute the original adjoint-gradient formula with JAX arrays.

    This intentionally mirrors `mode2_eps_then_sig.gradient.compute_gradient`
    instead of autodiffing the forward model. The goal is to compare
    and eventually accelerate the gradient convention already used by
    `gpr-inversion`.
    """
    if not is_jax_available():
        raise RuntimeError("JAX is not installed in this Python environment.")

    import jax
    import jax.numpy as jnp

    static_params = _build_static_jax_params(case, jnp)
    params = _with_material_params(static_params, epsilon, sigma, case, jnp)
    _data, forward_wavefield = forward_jax_array(
        case,
        epsilon,
        sigma,
        return_wavefield=True,
    )
    receivers = jnp.asarray(
        [(rec_x + case.npml, rec_z + case.npml) for rec_x, rec_z in case.receivers],
        dtype=jnp.int32,
    )
    reverse_params = dict(params)
    reverse_params["ca"] = params["ca_r"]

    single_shot_gradient = _single_shot_adjoint_gradient(
        jax,
        jnp,
        reverse_params,
        receivers,
        case,
    )
    grad_eps_shots, grad_sig_shots = jax.vmap(single_shot_gradient)(
        forward_wavefield,
        residual,
    )
    grad_eps = jnp.sum(grad_eps_shots, axis=0)
    grad_sig = jnp.sum(grad_sig_shots, axis=0)

    mask = jnp.ones((case.epsilon.shape[0], case.epsilon.shape[1]), dtype=jnp.float32)
    mask = mask.at[:10, :].set(0.0)
    grad_eps = grad_eps * mask
    grad_sig = grad_sig * mask

    if sigma_required_gradient:
        return grad_eps, grad_sig
    return grad_eps, None


def _single_shot_adjoint_gradient(jax: Any, jnp: Any, params: dict[str, Any], receivers, case: TinyFdtdCase):
    def single_shot(forward_wavefield, shot_residual):
        adjoint = _run_reverse_time_loop(jax, jnp, params, receivers, shot_residual)
        adjoint_forward_order = jnp.flip(adjoint, axis=0)

        npml = case.npml
        xl, zl = case.epsilon.shape
        x_slice = slice(npml, npml + xl)
        z_slice = slice(npml, npml + zl)

        forward_diff = (forward_wavefield[2:] - forward_wavefield[:-2]) / (2.0 * case.dt)
        adjoint_mid = adjoint_forward_order[1:case.steps - 1, x_slice, z_slice]
        forward_diff_inner = forward_diff[:, x_slice, z_slice]
        forward_mid = forward_wavefield[1:case.steps - 1, x_slice, z_slice]

        ep0 = jnp.asarray(8.841941282883074e-12, dtype=jnp.float32)
        grad_eps = ep0 * jnp.sum(adjoint_mid * forward_diff_inner, axis=0)
        grad_sig = jnp.sum(adjoint_mid * forward_mid, axis=0)
        return grad_eps, grad_sig

    return single_shot


def _run_reverse_time_loop(jax: Any, jnp: Any, params: dict[str, Any], receivers, shot_residual):
    nx = params["nx"]
    nz = params["nz"]
    npml = params["npml"]

    ey = jnp.zeros((nx, nz), dtype=jnp.float32)
    hz = jnp.zeros((nx, nz), dtype=jnp.float32)
    hx = jnp.zeros((nx, nz), dtype=jnp.float32)
    mem_dey_dx = jnp.zeros((2 * npml, nz), dtype=jnp.float32)
    mem_dey_dz = jnp.zeros((nx, 2 * npml), dtype=jnp.float32)
    mem_dhz_dx = jnp.zeros((2 * npml, nz), dtype=jnp.float32)
    mem_dhx_dz = jnp.zeros((nx, 2 * npml), dtype=jnp.float32)

    residual_reversed = jnp.flip(shot_residual, axis=1).T

    def step(carry, receiver_values):
        ey, hz, hx, mem_dey_dx, mem_dey_dz, mem_dhz_dx, mem_dhx_dz = carry
        ey = ey.at[receivers[:, 0], receivers[:, 1]].add(-receiver_values)
        hz, hx, mem_dey_dx, mem_dey_dz = _update_h(
            ey,
            hz,
            hx,
            mem_dey_dx,
            mem_dey_dz,
            params,
            jnp,
        )
        ey, mem_dhz_dx, mem_dhx_dz = _update_e(
            ey,
            hz,
            hx,
            mem_dhz_dx,
            mem_dhx_dz,
            params,
            jnp,
        )
        return (ey, hz, hx, mem_dey_dx, mem_dey_dz, mem_dhz_dx, mem_dhx_dz), ey

    carry = (ey, hz, hx, mem_dey_dx, mem_dey_dz, mem_dhz_dx, mem_dhx_dz)
    _final_carry, adjoint_series = jax.lax.scan(step, carry, residual_reversed)
    return adjoint_series
