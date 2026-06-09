"""Tiny JAX gradient checks for the FDTD lab."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cases import TinyFdtdCase, build_tiny_case
from .jax_adjoint import adjoint_gradient_jax_array
from .jax_forward import forward_jax_array, is_jax_available
from .losses import LossNorm, data_misfit_loss, l1_data_loss


@dataclass(frozen=True)
class DirectionalGradientCheck:
    """Result of a finite-difference directional derivative check."""

    loss: float
    analytic_directional_derivative: float
    finite_difference_derivative: float
    relative_error: float
    finite: bool
    loss_norm: str = "l2"


@dataclass(frozen=True)
class ResidualBackwardCheck:
    """Result of checking direct residual-loss backward behavior."""

    loss: float
    gradient: tuple[float, ...]
    expected_gradient: tuple[float, ...]
    max_abs_error: float
    finite: bool
    zero_subgradient: bool


@dataclass(frozen=True)
class NetworkSigmaGradientCheck:
    """Gradient check for a small JAX sigma parameterization."""

    loss: float
    analytic_directional_derivative: float
    finite_difference_derivative: float
    relative_error: float
    finite: bool
    loss_norm: str


def run_epsilon_directional_gradient_check(
    case: TinyFdtdCase | None = None,
    perturbation_scale: float = 0.05,
    finite_difference_step: float = 1e-2,
    loss_norm: LossNorm = "l2",
) -> DirectionalGradientCheck:
    """Compare JAX epsilon gradient with a finite-difference directional derivative."""
    if not is_jax_available():
        raise RuntimeError("JAX is not installed in this Python environment.")

    import jax
    import jax.numpy as jnp

    if case is None:
        case = build_tiny_case(steps=8, dtype=np.float32)

    epsilon0 = jnp.asarray(case.epsilon, dtype=jnp.float32)
    sigma0 = jnp.asarray(case.sigma, dtype=jnp.float32)
    sigma0 = jnp.asarray(case.sigma, dtype=jnp.float32)
    direction = _apply_fixed_top_mask(_build_smooth_direction(case, jnp), jnp)
    target_data = jax.lax.stop_gradient(
        forward_jax_array(case, epsilon0 + perturbation_scale * direction, sigma0)
    )

    def loss_fn(epsilon):
        data = forward_jax_array(case, epsilon, sigma0)
        return data_misfit_loss(data, target_data, jnp, loss_norm=loss_norm)

    predicted_data = forward_jax_array(case, epsilon0, sigma0)
    loss_value = data_misfit_loss(predicted_data, target_data, jnp, loss_norm=loss_norm)
    adjoint_source = _loss_adjoint_source(predicted_data, target_data, jnp, loss_norm)
    gradient, _grad_sig = adjoint_gradient_jax_array(
        case,
        epsilon0,
        sigma0,
        adjoint_source,
        sigma_required_gradient=False,
    )
    analytic = jnp.sum(gradient * direction)
    h = finite_difference_step
    loss_plus = loss_fn(epsilon0 + h * direction)
    loss_minus = loss_fn(epsilon0 - h * direction)
    finite_difference = (loss_plus - loss_minus) / (2.0 * h)
    denom = jnp.maximum(
        jnp.maximum(jnp.abs(analytic), jnp.abs(finite_difference)),
        jnp.asarray(1e-12, dtype=jnp.float32),
    )
    relative_error = jnp.abs(analytic - finite_difference) / denom
    values = jnp.asarray([loss_value, analytic, finite_difference, relative_error])

    return DirectionalGradientCheck(
        loss=float(loss_value),
        analytic_directional_derivative=float(analytic),
        finite_difference_derivative=float(finite_difference),
        relative_error=float(relative_error),
        finite=bool(np.isfinite(np.asarray(values)).all()),
        loss_norm=loss_norm,
    )


def run_sigma_directional_gradient_check(
    case: TinyFdtdCase | None = None,
    perturbation_scale: float = 5e-4,
    finite_difference_step: float = 2e-4,
    loss_norm: LossNorm = "l2",
) -> DirectionalGradientCheck:
    """Compare JAX sigma gradient with a finite-difference directional derivative."""
    if not is_jax_available():
        raise RuntimeError("JAX is not installed in this Python environment.")

    import jax
    import jax.numpy as jnp

    if case is None:
        case = build_tiny_case(steps=8, dtype=np.float32)

    epsilon0 = jnp.asarray(case.epsilon, dtype=jnp.float32)
    sigma0 = jnp.asarray(case.sigma, dtype=jnp.float32)
    direction = _apply_fixed_top_mask(_build_smooth_direction(case, jnp), jnp)
    target_data = jax.lax.stop_gradient(
        forward_jax_array(case, epsilon0, sigma0 + perturbation_scale * direction)
    )

    def loss_fn(sigma):
        data = forward_jax_array(case, epsilon0, sigma)
        return data_misfit_loss(data, target_data, jnp, loss_norm=loss_norm)

    predicted_data = forward_jax_array(case, epsilon0, sigma0)
    loss_value = data_misfit_loss(predicted_data, target_data, jnp, loss_norm=loss_norm)
    adjoint_source = _loss_adjoint_source(predicted_data, target_data, jnp, loss_norm)
    _grad_eps, gradient = adjoint_gradient_jax_array(
        case,
        epsilon0,
        sigma0,
        adjoint_source,
        sigma_required_gradient=True,
    )
    analytic = jnp.sum(gradient * direction)
    h = finite_difference_step
    loss_plus = loss_fn(sigma0 + h * direction)
    loss_minus = loss_fn(sigma0 - h * direction)
    finite_difference = (loss_plus - loss_minus) / (2.0 * h)
    denom = jnp.maximum(
        jnp.maximum(jnp.abs(analytic), jnp.abs(finite_difference)),
        jnp.asarray(1e-12, dtype=jnp.float32),
    )
    relative_error = jnp.abs(analytic - finite_difference) / denom
    values = jnp.asarray([loss_value, analytic, finite_difference, relative_error])

    return DirectionalGradientCheck(
        loss=float(loss_value),
        analytic_directional_derivative=float(analytic),
        finite_difference_derivative=float(finite_difference),
        relative_error=float(relative_error),
        finite=bool(np.isfinite(np.asarray(values)).all()),
        loss_norm=loss_norm,
    )


def run_l1_residual_backward_check(
    zero_subgradient: bool = True,
) -> ResidualBackwardCheck:
    """Check the direct backward rule used by mean L1 residual loss."""
    if not is_jax_available():
        raise RuntimeError("JAX is not installed in this Python environment.")

    import jax
    import jax.numpy as jnp

    residual = jnp.asarray([-2.0, -0.5, 0.0, 0.25, 3.0], dtype=jnp.float32)
    observed = jnp.zeros_like(residual)

    loss_value = l1_data_loss(
        residual,
        observed,
        jnp,
        zero_subgradient=zero_subgradient,
    )
    gradient = _loss_adjoint_source(
        residual,
        observed,
        jnp,
        loss_norm="l1",
        zero_l1_subgradient=zero_subgradient,
    )
    expected = jnp.sign(residual) / residual.size
    if not zero_subgradient:
        expected = expected.at[residual == 0.0].set(1.0 / residual.size)
    max_abs_error = jnp.max(jnp.abs(gradient - expected))
    values = jnp.concatenate(
        [
            jnp.asarray([loss_value, max_abs_error], dtype=jnp.float32),
            gradient,
            expected,
        ]
    )
    return ResidualBackwardCheck(
        loss=float(loss_value),
        gradient=tuple(float(value) for value in np.asarray(gradient)),
        expected_gradient=tuple(float(value) for value in np.asarray(expected)),
        max_abs_error=float(max_abs_error),
        finite=bool(np.isfinite(np.asarray(values)).all()),
        zero_subgradient=zero_subgradient,
    )


def run_network_sigma_directional_gradient_check(
    case: TinyFdtdCase | None = None,
    finite_difference_step: float = 1e-3,
    loss_norm: LossNorm = "l2",
) -> NetworkSigmaGradientCheck:
    """Check that loss gradients flow through a JAX sigma parameterization."""
    if not is_jax_available():
        raise RuntimeError("JAX is not installed in this Python environment.")

    import jax
    import jax.numpy as jnp

    if case is None:
        case = build_tiny_case(steps=8, dtype=np.float32)

    epsilon0 = jnp.asarray(case.epsilon, dtype=jnp.float32)
    sigma0 = jnp.asarray(case.sigma, dtype=jnp.float32)
    params0 = _build_tiny_sigma_network_params(jnp)
    direction = {
        "w_eps": jnp.asarray(0.31, dtype=jnp.float32),
        "w_depth": jnp.asarray(-0.17, dtype=jnp.float32),
        "w_hidden": jnp.asarray(0.23, dtype=jnp.float32),
        "bias": jnp.asarray(0.11, dtype=jnp.float32),
    }
    target_params = {
        "w_eps": params0["w_eps"] + jnp.asarray(0.07, dtype=jnp.float32),
        "w_depth": params0["w_depth"] - jnp.asarray(0.04, dtype=jnp.float32),
        "w_hidden": params0["w_hidden"] + jnp.asarray(0.05, dtype=jnp.float32),
        "bias": params0["bias"] - jnp.asarray(0.03, dtype=jnp.float32),
    }
    target_sigma = _apply_fixed_top_model(
        _tiny_sigma_network(target_params, epsilon0, jnp),
        sigma0,
        jnp,
    )
    target_data = jax.lax.stop_gradient(forward_jax_array(case, epsilon0, target_sigma))

    def loss_fn(params):
        sigma = _apply_fixed_top_model(
            _tiny_sigma_network(params, epsilon0, jnp),
            sigma0,
            jnp,
        )
        data = forward_jax_array(case, epsilon0, sigma)
        return data_misfit_loss(data, target_data, jnp, loss_norm=loss_norm)

    sigma_predicted = _apply_fixed_top_model(
        _tiny_sigma_network(params0, epsilon0, jnp),
        sigma0,
        jnp,
    )
    predicted_data = forward_jax_array(case, epsilon0, sigma_predicted)
    loss_value = data_misfit_loss(predicted_data, target_data, jnp, loss_norm=loss_norm)
    adjoint_source = _loss_adjoint_source(predicted_data, target_data, jnp, loss_norm)
    _grad_eps, sigma_gradient = adjoint_gradient_jax_array(
        case,
        epsilon0,
        sigma_predicted,
        adjoint_source,
        sigma_required_gradient=True,
    )
    h = finite_difference_step
    sigma_plus = _apply_fixed_top_model(
        _tiny_sigma_network(_tree_add_scaled(params0, direction, h, jax), epsilon0, jnp),
        sigma0,
        jnp,
    )
    sigma_minus = _apply_fixed_top_model(
        _tiny_sigma_network(_tree_add_scaled(params0, direction, -h, jax), epsilon0, jnp),
        sigma0,
        jnp,
    )
    sigma_directional_derivative = (sigma_plus - sigma_minus) / (2.0 * h)
    analytic = jnp.sum(sigma_gradient * sigma_directional_derivative)
    h = finite_difference_step
    loss_plus = loss_fn(_tree_add_scaled(params0, direction, h, jax))
    loss_minus = loss_fn(_tree_add_scaled(params0, direction, -h, jax))
    finite_difference = (loss_plus - loss_minus) / (2.0 * h)
    denom = jnp.maximum(
        jnp.maximum(jnp.abs(analytic), jnp.abs(finite_difference)),
        jnp.asarray(1e-12, dtype=jnp.float32),
    )
    relative_error = jnp.abs(analytic - finite_difference) / denom
    values = jnp.concatenate(
        [
            jnp.asarray([loss_value, analytic, finite_difference, relative_error], dtype=jnp.float32),
            jnp.ravel(sigma_gradient),
            jnp.ravel(sigma_directional_derivative),
        ]
    )

    return NetworkSigmaGradientCheck(
        loss=float(loss_value),
        analytic_directional_derivative=float(analytic),
        finite_difference_derivative=float(finite_difference),
        relative_error=float(relative_error),
        finite=bool(np.isfinite(np.asarray(values)).all()),
        loss_norm=loss_norm,
    )


def _build_smooth_direction(case: TinyFdtdCase, jnp):
    nx, nz = case.epsilon.shape
    x = jnp.linspace(0.0, 1.0, nx)[:, None]
    z = jnp.linspace(0.0, 1.0, nz)[None, :]
    direction = jnp.sin(jnp.pi * x) * jnp.cos(2.0 * jnp.pi * z)
    max_abs = jnp.max(jnp.abs(direction))
    return direction / jnp.maximum(max_abs, 1e-12)


def _loss_adjoint_source(
    predicted,
    observed,
    jnp,
    loss_norm: LossNorm,
    zero_l1_subgradient: bool = True,
):
    """Return d(loss)/d(predicted) explicitly, without JAX autodiff."""
    residual = predicted - observed
    scale = jnp.asarray(1.0 / residual.size, dtype=residual.dtype)
    if loss_norm == "l2":
        return 2.0 * residual * scale
    if loss_norm == "l1":
        source = jnp.sign(residual) * scale
        if zero_l1_subgradient:
            source = jnp.where(residual == 0.0, 0.0, source)
        else:
            source = jnp.where(residual == 0.0, scale, source)
        return source
    raise ValueError(f"Unsupported loss_norm: {loss_norm!r}")


def _apply_fixed_top_mask(values, jnp, fixed_top_rows: int = 10):
    mask = jnp.ones_like(values)
    return mask.at[:fixed_top_rows, :].set(0.0) * values


def _apply_fixed_top_model(values, fixed_reference, jnp, fixed_top_rows: int = 10):
    trainable_mask = jnp.ones_like(values).at[:fixed_top_rows, :].set(0.0)
    fixed_mask = jnp.ones_like(values) - trainable_mask
    return values * trainable_mask + fixed_reference * fixed_mask


def _build_tiny_sigma_network_params(jnp):
    return {
        "w_eps": jnp.asarray(0.8, dtype=jnp.float32),
        "w_depth": jnp.asarray(-0.35, dtype=jnp.float32),
        "w_hidden": jnp.asarray(0.5, dtype=jnp.float32),
        "bias": jnp.asarray(-0.1, dtype=jnp.float32),
    }


def _tiny_sigma_network(params, epsilon, jnp):
    nx, _nz = epsilon.shape
    eps_norm = (epsilon - jnp.mean(epsilon)) / (jnp.std(epsilon) + 1e-6)
    depth = jnp.linspace(-1.0, 1.0, nx, dtype=jnp.float32)[:, None]
    hidden = jnp.tanh(params["w_eps"] * eps_norm + params["w_depth"] * depth)
    logits = params["w_hidden"] * hidden + params["bias"]
    min_sig = jnp.asarray(3.0e-3, dtype=jnp.float32)
    max_sig = jnp.asarray(8.5e-3, dtype=jnp.float32)
    return min_sig + (max_sig - min_sig) * (1.0 / (1.0 + jnp.exp(-logits)))


def _tree_add_scaled(params, direction, scale, jax):
    return jax.tree_util.tree_map(lambda value, step: value + scale * step, params, direction)


def _tree_dot(left, right, jax, jnp):
    products = jax.tree_util.tree_map(lambda lhs, rhs: jnp.sum(lhs * rhs), left, right)
    leaves = jax.tree_util.tree_leaves(products)
    return sum(leaves, jnp.asarray(0.0, dtype=jnp.float32))
