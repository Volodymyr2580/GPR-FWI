"""Optional JAX prototype for the 2D TEz FDTD forward model."""

from __future__ import annotations

import importlib.util
import os
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from gpr_inversion.common.Add_CPML import Add_CPML
from gpr_inversion.common.Wavelet import ricker

from .cases import TinyFdtdCase


@dataclass(frozen=True)
class JaxEnvironment:
    """JAX availability and device information for progress records."""

    available: bool
    version: str | None
    devices: tuple[str, ...]
    default_backend: str | None


def is_jax_available() -> bool:
    """Return whether the active Python environment can import JAX."""
    return importlib.util.find_spec("jax") is not None


def _case_static_key(case: TinyFdtdCase) -> tuple[Any, ...]:
    return (
        tuple(case.epsilon.shape),
        tuple(case.sigma.shape),
        tuple(case.sources),
        tuple(case.receivers),
        float(case.dt),
        float(case.dx),
        float(case.dz),
        int(case.npml),
        float(case.freq),
        int(case.steps),
    )


def describe_jax_environment() -> JaxEnvironment:
    """Describe the active JAX environment without requiring callers to import JAX."""
    if not is_jax_available():
        return JaxEnvironment(
            available=False,
            version=None,
            devices=(),
            default_backend=None,
        )

    import jax

    return JaxEnvironment(
        available=True,
        version=jax.__version__,
        devices=tuple(str(device) for device in jax.devices()),
        default_backend=jax.default_backend(),
    )


def _jax_default_device_context(jax: Any):
    requested = os.environ.get("JAX_FDTD_DEVICE", "").strip()
    if not requested:
        return nullcontext()

    normalized = requested.lower().replace("cuda", "gpu")
    if ":" in normalized:
        platform, index_text = normalized.split(":", 1)
        index = int(index_text)
    else:
        platform = normalized
        index = 0

    devices = jax.devices(platform)
    if index >= len(devices):
        raise RuntimeError(
            f"Requested JAX_FDTD_DEVICE={requested!r}, but JAX only sees "
            f"{len(devices)} {platform!r} device(s): {devices}"
        )
    return jax.default_device(devices[index])


def _is_resource_exhausted_error(exc: Exception) -> bool:
    text = str(exc).upper()
    return "RESOURCE_EXHAUSTED" in text or "OUT_OF_MEMORY" in text or "CUDA_ERROR_OUT_OF_MEMORY" in text


def _clear_jax_caches() -> None:
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass


def run_jax_forward(case: TinyFdtdCase) -> np.ndarray:
    """Run the JAX forward prototype and return NumPy receiver data.

    The time loop is implemented with `jax.lax.scan`, so once JAX is installed
    this function exercises the intended GPU-friendly control-flow structure.
    """
    if not is_jax_available():
        raise RuntimeError(
            "JAX is not installed in this Python environment. "
            "Use `conda run -n gprfwi python -c \"import jax\"` to verify setup."
        )

    import jax
    import jax.numpy as jnp

    with _jax_default_device_context(jax):
        epsilon = jnp.asarray(case.epsilon, dtype=jnp.float32)
        sigma = jnp.asarray(case.sigma, dtype=jnp.float32)
        data = forward_jax_array(case, epsilon, sigma)
    return np.asarray(jax.device_get(data))


def run_jax_forward_with_wavefield(case: TinyFdtdCase) -> tuple[np.ndarray, np.ndarray]:
    """Run the JAX forward prototype and return receiver data plus Ey wavefields."""
    if not is_jax_available():
        raise RuntimeError(
            "JAX is not installed in this Python environment. "
            "Use `conda run -n gprfwi python -c \"import jax\"` to verify setup."
        )

    import jax
    import jax.numpy as jnp

    with _jax_default_device_context(jax):
        epsilon = jnp.asarray(case.epsilon, dtype=jnp.float32)
        sigma = jnp.asarray(case.sigma, dtype=jnp.float32)
        data, wavefield = forward_jax_array(case, epsilon, sigma, return_wavefield=True)
    return np.asarray(jax.device_get(data)), np.asarray(jax.device_get(wavefield))


def run_jax_forward_with_illumination(case: TinyFdtdCase) -> tuple[np.ndarray, np.ndarray]:
    """Run the JAX forward prototype and return receiver data plus illumination."""
    if not is_jax_available():
        raise RuntimeError(
            "JAX is not installed in this Python environment. "
            "Use `conda run -n gprfwi python -c \"import jax\"` to verify setup."
        )

    import jax
    import jax.numpy as jnp

    with _jax_default_device_context(jax):
        epsilon = jnp.asarray(case.epsilon, dtype=jnp.float32)
        sigma = jnp.asarray(case.sigma, dtype=jnp.float32)
        data, illumination = forward_jax_array(case, epsilon, sigma, return_illumination=True)
    return np.asarray(jax.device_get(data)), np.asarray(jax.device_get(illumination))


_JITTED_FORWARD_ILLUMINATION_CACHE: dict[tuple[Any, ...], Any] = {}


def run_jax_forward_with_illumination_jitted(case: TinyFdtdCase) -> tuple[np.ndarray, np.ndarray]:
    """Run forward plus illumination through a cached JIT kernel."""
    if not is_jax_available():
        raise RuntimeError(
            "JAX is not installed in this Python environment. "
            "Use `conda run -n gprfwi python -c \"import jax\"` to verify setup."
        )

    import jax
    import jax.numpy as jnp

    key = ("forward_illumination", _case_static_key(case))
    with _jax_default_device_context(jax):
        fn = _JITTED_FORWARD_ILLUMINATION_CACHE.get(key)
        if fn is None:
            fn = jax.jit(
                lambda eps, sig: forward_jax_array(
                    case,
                    eps,
                    sig,
                    return_illumination=True,
                )
            )
            _JITTED_FORWARD_ILLUMINATION_CACHE[key] = fn
        epsilon = jnp.asarray(case.epsilon, dtype=jnp.float32)
        sigma = jnp.asarray(case.sigma, dtype=jnp.float32)
        data, illumination = fn(epsilon, sigma)
    return np.asarray(jax.device_get(data)), np.asarray(jax.device_get(illumination))


def run_jax_forward_with_illumination_batched(
    case: TinyFdtdCase,
    shot_batch_size: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run JAX forward plus illumination in source batches.

    Illumination needs every time-step field squared and accumulated. Running
    all shots in one `vmap` can require a large temporary tensor, so production
    training calls this batched wrapper by default.
    """
    n_shots = len(case.sources)
    if n_shots == 0:
        return (
            np.zeros((0, len(case.receivers), case.steps), dtype=np.float32),
            np.zeros_like(case.epsilon, dtype=np.float32),
        )
    if shot_batch_size <= 0 or shot_batch_size >= n_shots:
        return run_jax_forward_with_illumination(case)

    data_parts: list[np.ndarray] = []
    illumination_total = np.zeros_like(case.epsilon, dtype=np.float32)
    start = 0
    current_batch_size = min(shot_batch_size, n_shots)
    while start < n_shots:
        current_batch_size = min(current_batch_size, n_shots - start)
        while True:
            stop = start + current_batch_size
            batch_case = replace(case, sources=case.sources[start:stop])
            try:
                batch_data, batch_illumination = run_jax_forward_with_illumination_jitted(batch_case)
                break
            except Exception as exc:
                if current_batch_size <= 1 or not _is_resource_exhausted_error(exc):
                    raise
                _clear_jax_caches()
                current_batch_size = max(1, current_batch_size // 2)
                print(
                    "JAX forward illumination OOM; retrying with "
                    f"shot_batch_size={current_batch_size}",
                    flush=True,
                )
        data_parts.append(np.asarray(batch_data, dtype=np.float32))
        illumination_total += np.asarray(batch_illumination, dtype=np.float32)
        start += current_batch_size
    return np.concatenate(data_parts, axis=0), illumination_total


def forward_jax_array(
    case: TinyFdtdCase,
    epsilon: Any,
    sigma: Any,
    return_wavefield: bool = False,
    return_illumination: bool = False,
):
    """Run the JAX forward prototype and return a JAX array.

    Unlike `run_jax_forward`, this function keeps the result on the JAX side and
    can be used by JIT-compiled forward and manual-adjoint experiments. Model
    gradients in this project should use `jax_adjoint.adjoint_gradient_jax_array`
    rather than differentiating this forward computation graph directly.
    """
    if not is_jax_available():
        raise RuntimeError("JAX is not installed in this Python environment.")

    import jax
    import jax.numpy as jnp

    static_params = _build_static_jax_params(case, jnp)
    params = _with_material_params(static_params, epsilon, sigma, case, jnp)
    sources = jnp.asarray(
        [(src_x + case.npml, src_z + case.npml) for src_x, src_z in case.sources],
        dtype=jnp.int32,
    )
    receivers = jnp.asarray(
        [(rec_x + case.npml, rec_z + case.npml) for rec_x, rec_z in case.receivers],
        dtype=jnp.int32,
    )
    time_axis = np.arange(case.steps, dtype=np.float64) * case.dt
    wavelet = jnp.asarray(ricker(time_axis, case.freq), dtype=jnp.float32)

    single_shot = _single_shot_forward(
        jax,
        jnp,
        params,
        return_wavefield=return_wavefield,
        return_illumination=return_illumination,
    )
    data = jax.vmap(lambda source: single_shot(source, receivers, wavelet))(sources)
    if return_wavefield and return_illumination:
        receiver_data, wavefield_data, illumination_by_shot = data
        return receiver_data, wavefield_data, jnp.sum(illumination_by_shot, axis=0)
    if return_wavefield:
        receiver_data, wavefield_data = data
        return receiver_data, wavefield_data
    if return_illumination:
        receiver_data, illumination_by_shot = data
        return receiver_data, jnp.sum(illumination_by_shot, axis=0)
    return data


def _build_static_jax_params(case: TinyFdtdCase, jnp: Any) -> dict[str, Any]:
    xl, zl = case.epsilon.shape
    epsilon_extended = np.pad(case.epsilon, ((case.npml, case.npml), (case.npml, case.npml)), "edge")
    sigma_extended = np.pad(case.sigma, ((case.npml, case.npml), (case.npml, case.npml)), "edge")
    mu_extended = np.ones((xl + 2 * case.npml, zl + 2 * case.npml), dtype=case.epsilon.dtype)
    cpml = Add_CPML(
        xl,
        zl,
        sigma_extended.copy(),
        epsilon_extended.copy(),
        mu_extended.copy(),
        case.dx,
        case.dz,
        case.dt,
    )

    return {
        "xl": xl,
        "zl": zl,
        "nx": xl + 2 * case.npml,
        "nz": zl + 2 * case.npml,
        "dx": case.dx,
        "dz": case.dz,
        "dt": case.dt,
        "npml": case.npml,
        "a_x": jnp.asarray(cpml.a_x, dtype=jnp.float32),
        "b_x": jnp.asarray(cpml.b_x, dtype=jnp.float32),
        "k_x": jnp.asarray(cpml.k_x, dtype=jnp.float32),
        "a_z": jnp.asarray(cpml.a_z, dtype=jnp.float32),
        "b_z": jnp.asarray(cpml.b_z, dtype=jnp.float32),
        "k_z": jnp.asarray(cpml.k_z, dtype=jnp.float32),
        "a_x_half": jnp.asarray(cpml.a_x_half, dtype=jnp.float32),
        "b_x_half": jnp.asarray(cpml.b_x_half, dtype=jnp.float32),
        "k_x_half": jnp.asarray(cpml.k_x_half, dtype=jnp.float32),
        "a_z_half": jnp.asarray(cpml.a_z_half, dtype=jnp.float32),
        "b_z_half": jnp.asarray(cpml.b_z_half, dtype=jnp.float32),
        "k_z_half": jnp.asarray(cpml.k_z_half, dtype=jnp.float32),
        "mu": jnp.asarray(mu_extended * 1.2566370614359173e-06, dtype=jnp.float32),
    }


def _with_material_params(static_params: dict[str, Any], epsilon, sigma, case: TinyFdtdCase, jnp: Any) -> dict[str, Any]:
    ep0 = 8.841941282883074e-12
    epsilon_extended = jnp.pad(epsilon, ((case.npml, case.npml), (case.npml, case.npml)), mode="edge")
    sigma_extended = jnp.pad(sigma, ((case.npml, case.npml), (case.npml, case.npml)), mode="edge")
    epsilon_physical = epsilon_extended * ep0
    denom = 1.0 + sigma_extended * case.dt / (2.0 * epsilon_physical)
    ca = (1.0 - sigma_extended * case.dt / (2.0 * epsilon_physical)) / denom
    cb = (1.0 / epsilon_physical) / denom
    ca_r = (epsilon_physical * 2.0) / (epsilon_physical * 2.0 + sigma_extended * case.dt)
    params = dict(static_params)
    params["ca"] = ca.astype(jnp.float32)
    params["cb"] = cb.astype(jnp.float32)
    params["ca_r"] = ca_r.astype(jnp.float32)
    return params


def _single_shot_forward(
    jax: Any,
    jnp: Any,
    params: dict[str, Any],
    return_wavefield: bool = False,
    return_illumination: bool = False,
):
    def single_shot(source, receivers, wavelet):
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

        def step(carry, source_value):
            ey, hz, hx, mem_dey_dx, mem_dey_dz, mem_dhz_dx, mem_dhx_dz = carry
            hz, hx, mem_dey_dx, mem_dey_dz = _update_h(
                ey, hz, hx, mem_dey_dx, mem_dey_dz, params, jnp
            )
            ey, mem_dhz_dx, mem_dhx_dz = _update_e(
                ey, hz, hx, mem_dhz_dx, mem_dhx_dz, params, jnp
            )
            src_x = source[0]
            src_z = source[1]
            source_term = -params["cb"][src_x, src_z] * source_value * params["dt"] / params["dx"] / params["dz"]
            ey = ey.at[src_x, src_z].add(source_term)
            receiver_data = ey[receivers[:, 0], receivers[:, 1]]
            outputs = [receiver_data]
            if return_wavefield:
                outputs.append(ey)
            if return_illumination:
                npml = params["npml"]
                xl = params["xl"]
                zl = params["zl"]
                ey_valid = ey[npml:npml + xl, npml:npml + zl]
                outputs.append(ey_valid**2)
            if len(outputs) > 1:
                return (
                    ey,
                    hz,
                    hx,
                    mem_dey_dx,
                    mem_dey_dz,
                    mem_dhz_dx,
                    mem_dhx_dz,
                ), tuple(outputs)
            return (ey, hz, hx, mem_dey_dx, mem_dey_dz, mem_dhz_dx, mem_dhx_dz), receiver_data

        carry = (ey, hz, hx, mem_dey_dx, mem_dey_dz, mem_dhz_dx, mem_dhx_dz)
        _final_carry, scan_output = jax.lax.scan(step, carry, wavelet)
        if return_wavefield and return_illumination:
            receiver_series, wavefield_series, illumination_series = scan_output
            return receiver_series.T, wavefield_series, jnp.sum(illumination_series, axis=0)
        if return_wavefield:
            receiver_series, wavefield_series = scan_output
            return receiver_series.T, wavefield_series
        if return_illumination:
            receiver_series, illumination_series = scan_output
            return receiver_series.T, jnp.sum(illumination_series, axis=0)
        receiver_series = scan_output
        return receiver_series.T

    return single_shot


def _update_h(ey, hz, hx, mem_dey_dx, mem_dey_dz, params, jnp):
    dx = params["dx"]
    dz = params["dz"]
    dt = params["dt"]
    npml = params["npml"]
    xl = params["xl"]
    nx = params["nx"]
    nz = params["nz"]
    mu = params["mu"]

    d_ey_dx = (ey[2:nx, 1:nz - 1] - ey[1:nx - 1, 1:nz - 1]) / dx
    d_ey_dz = (ey[1:nx - 1, 2:nz] - ey[1:nx - 1, 1:nz - 1]) / dz

    hz_eff = d_ey_dx
    left_i = jnp.arange(1, npml)
    right_i = jnp.arange(nx - npml, nx - 1)
    all_j = jnp.arange(1, nz - 1)

    left_val = d_ey_dx[0:npml - 1, :]
    left_mem = mem_dey_dx[left_i[:, None], all_j[None, :]]
    left_mem = params["b_x_half"][left_i][:, None] * left_mem + params["a_x_half"][left_i][:, None] * left_val
    mem_dey_dx = mem_dey_dx.at[left_i[:, None], all_j[None, :]].set(left_mem)
    hz_eff = hz_eff.at[0:npml - 1, :].set(left_val / params["k_x_half"][left_i][:, None] + left_mem)

    right_rows = right_i - 1
    right_mem_rows = right_i - xl
    right_val = d_ey_dx[right_rows[:, None], (all_j - 1)[None, :]]
    right_mem = mem_dey_dx[right_mem_rows[:, None], all_j[None, :]]
    right_mem = params["b_x_half"][right_i][:, None] * right_mem + params["a_x_half"][right_i][:, None] * right_val
    mem_dey_dx = mem_dey_dx.at[right_mem_rows[:, None], all_j[None, :]].set(right_mem)
    hz_eff = hz_eff.at[right_rows[:, None], (all_j - 1)[None, :]].set(
        right_val / params["k_x_half"][right_i][:, None] + right_mem
    )

    hx_eff = d_ey_dz
    left_j = jnp.arange(1, npml)
    right_j = jnp.arange(nz - npml, nz - 1)
    all_i = jnp.arange(1, nx - 1)

    bottom_val = d_ey_dz[:, 0:npml - 1]
    bottom_mem = mem_dey_dz[all_i[:, None], left_j[None, :]]
    bottom_mem = params["b_z_half"][left_j][None, :] * bottom_mem + params["a_z_half"][left_j][None, :] * bottom_val
    mem_dey_dz = mem_dey_dz.at[all_i[:, None], left_j[None, :]].set(bottom_mem)
    hx_eff = hx_eff.at[:, 0:npml - 1].set(bottom_val / params["k_z_half"][left_j][None, :] + bottom_mem)

    top_cols = right_j - 1
    top_mem_cols = right_j - params["zl"]
    top_val = d_ey_dz[(all_i - 1)[:, None], top_cols[None, :]]
    top_mem = mem_dey_dz[all_i[:, None], top_mem_cols[None, :]]
    top_mem = params["b_z_half"][right_j][None, :] * top_mem + params["a_z_half"][right_j][None, :] * top_val
    mem_dey_dz = mem_dey_dz.at[all_i[:, None], top_mem_cols[None, :]].set(top_mem)
    hx_eff = hx_eff.at[(all_i - 1)[:, None], top_cols[None, :]].set(
        top_val / params["k_z_half"][right_j][None, :] + top_mem
    )

    hz = hz.at[1:nx - 1, 1:nz - 1].add(hz_eff * dt / mu[1:nx - 1, 1:nz - 1])
    hx = hx.at[1:nx - 1, 1:nz - 1].add(-hx_eff * dt / mu[1:nx - 1, 1:nz - 1])
    return hz, hx, mem_dey_dx, mem_dey_dz


def _update_e(ey, hz, hx, mem_dhz_dx, mem_dhx_dz, params, jnp):
    dx = params["dx"]
    dz = params["dz"]
    dt = params["dt"]
    npml = params["npml"]
    xl = params["xl"]
    nx = params["nx"]
    nz = params["nz"]

    d_hz_dx = (hz[1:nx - 1, 1:nz - 1] - hz[0:nx - 2, 1:nz - 1]) / dx
    d_hx_dz = (hx[1:nx - 1, 1:nz - 1] - hx[1:nx - 1, 0:nz - 2]) / dz

    dx_eff = d_hz_dx
    left_i = jnp.arange(1, npml)
    right_i = jnp.arange(nx - npml, nx - 1)
    all_j = jnp.arange(1, nz - 1)

    left_val = d_hz_dx[0:npml - 1, :]
    left_mem = mem_dhz_dx[left_i[:, None], all_j[None, :]]
    left_mem = params["b_x"][left_i][:, None] * left_mem + params["a_x"][left_i][:, None] * left_val
    mem_dhz_dx = mem_dhz_dx.at[left_i[:, None], all_j[None, :]].set(left_mem)
    dx_eff = dx_eff.at[0:npml - 1, :].set(left_val / params["k_x"][left_i][:, None] + left_mem)

    right_rows = right_i - 1
    right_mem_rows = right_i - xl
    right_val = d_hz_dx[right_rows[:, None], (all_j - 1)[None, :]]
    right_mem = mem_dhz_dx[right_mem_rows[:, None], all_j[None, :]]
    right_mem = params["b_x"][right_i][:, None] * right_mem + params["a_x"][right_i][:, None] * right_val
    mem_dhz_dx = mem_dhz_dx.at[right_mem_rows[:, None], all_j[None, :]].set(right_mem)
    dx_eff = dx_eff.at[right_rows[:, None], (all_j - 1)[None, :]].set(
        right_val / params["k_x"][right_i][:, None] + right_mem
    )

    dz_eff = d_hx_dz
    left_j = jnp.arange(1, npml)
    right_j = jnp.arange(nz - npml, nz - 1)
    all_i = jnp.arange(1, nx - 1)

    bottom_val = d_hx_dz[:, 0:npml - 1]
    bottom_mem = mem_dhx_dz[all_i[:, None], left_j[None, :]]
    bottom_mem = params["b_z"][left_j][None, :] * bottom_mem + params["a_z"][left_j][None, :] * bottom_val
    mem_dhx_dz = mem_dhx_dz.at[all_i[:, None], left_j[None, :]].set(bottom_mem)
    dz_eff = dz_eff.at[:, 0:npml - 1].set(bottom_val / params["k_z"][left_j][None, :] + bottom_mem)

    top_cols = right_j - 1
    top_mem_cols = right_j - params["zl"]
    top_val = d_hx_dz[(all_i - 1)[:, None], top_cols[None, :]]
    top_mem = mem_dhx_dz[all_i[:, None], top_mem_cols[None, :]]
    top_mem = params["b_z"][right_j][None, :] * top_mem + params["a_z"][right_j][None, :] * top_val
    mem_dhx_dz = mem_dhx_dz.at[all_i[:, None], top_mem_cols[None, :]].set(top_mem)
    dz_eff = dz_eff.at[(all_i - 1)[:, None], top_cols[None, :]].set(
        top_val / params["k_z"][right_j][None, :] + top_mem
    )

    ey_inner = params["ca"][1:nx - 1, 1:nz - 1] * ey[1:nx - 1, 1:nz - 1]
    ey_inner = ey_inner + params["cb"][1:nx - 1, 1:nz - 1] * (dx_eff - dz_eff) * dt
    ey = ey.at[1:nx - 1, 1:nz - 1].set(ey_inner)
    return ey, mem_dhz_dx, mem_dhx_dz
