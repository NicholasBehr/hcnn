import jax
import jax.numpy as jnp


def td_residuals(
    rewards: jax.Array, values: jax.Array, td_gamma: float
) -> jax.Array:
    """Compute the temporal difference residuals.

    Args:
        rewards (jax.Array): Rewards for each step,
            shape (..., horizon).
        values (jax.Array): Value estimates for each step,
            shape (..., horizon).
        td_gamma (float): Discount factor for the temporal difference.

    Returns:
        jax.Array: Temporal difference residuals,
            shape (..., horizon-1).
    """
    return rewards[..., :-1] + td_gamma * values[..., 1:] - values[..., :-1]


@jax.jit
def generalized_advantage_1(
    td_residuals: jax.Array,
    td_gamma: float,
    gae_lambda: float,
) -> jax.Array:
    """Compute the Generalized Advantage Estimation (GAE).

    Args:
        td_residuals (jax.Array): Temporal difference residuals,
            shape (..., horizon-1).
        td_gamma (float): Discount factor for the temporal difference.
        gae_lambda (float): Lambda parameter for GAE.

    Returns:
        jax.Array: Generalized advantage estimates,
            shape (...).
    """

    # Number of elements along the horizon dimension
    num_elements = td_residuals.shape[-1]

    # Standard discount factors:
    # (gamma*lambda)^0, (gamma*lambda)^1, ...
    discounts = (td_gamma * gae_lambda) ** jnp.arange(num_elements)

    # Powers for the cutoff correction term: N, N-1, ..., 1
    # e.g., if num_elements = 3, this is [3, 2, 1]
    powers_for_cutoff = jnp.arange(num_elements, 0, -1)

    # Cutoff correction term:
    # (1 - lambda^N), (1 - lambda^(N-1)), ..., (1 - lambda^1)
    cutoff_correction = 1.0 - (gae_lambda**powers_for_cutoff)

    # Element-wise product and sum.
    return jnp.sum(discounts * td_residuals * cutoff_correction, axis=-1)


@jax.jit
def generalized_advantage_t(
    td_residuals: jax.Array,
    td_gamma: float,
    gae_lambda: float,
) -> jax.Array:
    """Compute the Generalized Advantage Estimation (GAE)
    for all starting points with one or more steps to go.

    Args:
        td_residuals (jax.Array): Temporal difference residuals,
            shape (..., horizon-1).
        td_gamma (float): Discount factor for the temporal difference.
        gae_lambda (float): Lambda parameter for GAE.

    Returns:
        jax.Array: Generalized advantage estimates for each starting point,
            shape (..., horizon-1).
    """
    # Return early if the horizon is empty.
    if td_residuals.shape[-1] == 0:
        return td_residuals

    horizon_len = td_residuals.shape[-1]
    powers = jnp.arange(horizon_len, 0, -1)
    cutoff_correction = 1.0 - (gae_lambda**powers)

    weighted_residuals = td_residuals * cutoff_correction
    scan_factor = td_gamma * gae_lambda

    def scan_body(carry_advantage, weighted_residual_t):
        # Implements the recursive formula:
        # A_t = weighted_residual_t + (gamma * lambda) * A_{t+1}
        advantage_t = weighted_residual_t + scan_factor * carry_advantage
        return advantage_t, advantage_t

    # The initial carry for the scan is the advantage at the step after the last
    # residual, which is 0. For a batch, this is a vector of zeros.
    initial_carry = jnp.zeros_like(td_residuals[..., 0])

    # jax.lax.scan iterates over the first axis. We want to scan over the
    # time axis (the last axis). So we move the time axis to the front.
    weighted_residuals_time_first = jnp.moveaxis(weighted_residuals, -1, 0)

    _, advantages_time_first = jax.lax.scan(
        scan_body,
        initial_carry,
        weighted_residuals_time_first,
        reverse=True,
    )

    # Move the time axis back to its original position.
    advantages = jnp.moveaxis(advantages_time_first, 0, -1)

    return advantages
