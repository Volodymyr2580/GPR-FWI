"""JAX data-misfit losses used by local inversion sanity checks."""

from __future__ import annotations

from typing import Any, Literal


LossNorm = Literal["l1", "l2"]


def l1_data_loss(predicted: Any, observed: Any, jnp: Any, zero_subgradient: bool = True):
    """Return mean absolute residual with an optional zero subgradient at zero.

    JAX's default ``grad(abs(x))`` chooses ``+1`` at exactly ``x == 0`` in the
    checked environment. The explicit ``where`` branch gives the conventional
    zero subgradient at exact zero residuals, which is easier to reason about in
    waveform-inversion tests.
    """
    residual = predicted - observed
    if zero_subgradient:
        abs_residual = jnp.where(residual == 0.0, 0.0, jnp.abs(residual))
    else:
        abs_residual = jnp.abs(residual)
    return jnp.mean(abs_residual)


def l2_data_loss(predicted: Any, observed: Any, jnp: Any):
    """Return mean squared residual."""
    residual = predicted - observed
    return jnp.mean(residual**2)


def data_misfit_loss(
    predicted: Any,
    observed: Any,
    jnp: Any,
    loss_norm: LossNorm = "l2",
    zero_l1_subgradient: bool = True,
):
    """Dispatch a data misfit loss by name."""
    if loss_norm == "l2":
        return l2_data_loss(predicted, observed, jnp)
    if loss_norm == "l1":
        return l1_data_loss(
            predicted,
            observed,
            jnp,
            zero_subgradient=zero_l1_subgradient,
        )
    raise ValueError(f"Unsupported loss_norm: {loss_norm!r}")
