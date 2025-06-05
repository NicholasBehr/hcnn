import jax
import jax.numpy as jnp


def td_residuals(rewards: jax.Array, values: jax.Array, td_gamma: float) -> jax.Array:
    """Compute the temporal difference residuals."""
    return rewards[:-1] + td_gamma * values[1:] - values[:-1]


@jax.jit
def generalized_advantage_1(
    deltas: jax.Array,
    td_gamma: float,
    gae_lambda: float,
) -> jax.Array:
    """Compute the Generalized Advantage Estimation (GAE) for a given slice of TD residuals."""

    # Number of elements in the current slice of td_residuals
    num_elements = deltas.shape[0]
    if num_elements == 0:
        return jnp.array(0.0, dtype=jnp.result_type(float, deltas))

    # Standard discount factors: (gamma*lambda)^0, (gamma*lambda)^1, ...
    discounts = (td_gamma * gae_lambda) ** jnp.arange(0, num_elements)

    # Powers for the cutoff correction term: N, N-1, ..., 1
    # e.g., if num_elements = 3, this is [3, 2, 1]
    powers_for_cutoff = jnp.arange(num_elements, 0, -1)

    # Cutoff correction term: (1 - lambda^N), (1 - lambda^(N-1)), ..., (1 - lambda^1)
    cutoff_correction = 1.0 - (gae_lambda**powers_for_cutoff)

    # Element-wise product and sum
    return jnp.sum(discounts * deltas * cutoff_correction)


@jax.jit
def generalized_advantage_t(
    deltas: jax.Array,
    td_gamma: float,
    gae_lambda: float,
) -> jax.Array:
    """Compute the Generalized Advantage Estimation (GAE) for all starting points, vectorized."""

    if deltas.shape[0] == 0:
        return jnp.array([], dtype=jnp.result_type(float, deltas))

    # Step 1: Prepare the c_array for the scan.
    # c_j = deltas[j] * (1 - gae_lambda**(m-j))
    # The exponent (m-j) for gae_lambda corresponds to the number of terms
    # from index j to the end of the deltas array.
    # powers_lambda_exponent will be [m, m-1, ..., 1]
    powers_lambda_exponent = jnp.arange(deltas.shape[0], 0, -1)

    # cutoff_term_for_cj[j] evaluates to (1 - gae_lambda**(m-j))
    cutoff_term_for_cj = 1.0 - (gae_lambda**powers_lambda_exponent)

    c_array = deltas * cutoff_term_for_cj  # Element-wise product

    # Step 2: Perform a backward scan.
    # The recurrence relation for advantages a_s is:
    # a_s = sum_{k=0}^{m-s-1} (td_gamma*gae_lambda)^k * deltas[s+k] * (1 - gae_lambda^{m-s-k})
    # This can be rewritten as a_s = c_s + (td_gamma*gae_lambda) * a_{s+1} (if c included deltas)
    # More precisely, a_s = sum_{j=s}^{m-1} (td_gamma*gae_lambda)^{j-s} * (deltas[j] * (1 - gae_lambda^{m-j}))
    # Let c_j = deltas[j] * (1 - gae_lambda^{m-j}).
    # Then a_s = sum_{k=0}^{m-1-s} (td_gamma*gae_lambda)^k * c_{s+k}.
    # This is a_s = c_s + (td_gamma*gae_lambda) * c_{s+1} + (td_gamma*gae_lambda)^2 * c_{s+2} + ...
    # This can be computed with a backward scan: a_j = c_j + (td_gamma*gae_lambda) * a_{j+1}, with a_m = 0.

    scan_combination_factor = td_gamma * gae_lambda

    def scan_body(advantage_at_next_step, c_at_current_step):
        """
        Args:
            advantage_at_next_step (a_{j+1}): Carry from the previous scan step (scanning in reverse).
            c_at_current_step (c_j): Current element from c_array.
        Returns:
            Tuple (new_carry, y_element): (a_j, a_j)
        """
        # Calculate a_j = c_j + scan_factor * a_{j+1}
        advantage_at_current_step = c_at_current_step + scan_combination_factor * advantage_at_next_step

        # The new carry for the (j-1)th step is a_j.
        # The value to be collected for the j-th output is a_j.
        return advantage_at_current_step, advantage_at_current_step

    # initial_carry_for_scan represents a_m (advantage after the last delta term), which is 0.0.
    initial_carry_for_scan = 0.0

    # jax.lax.scan with reverse=True processes c_array from last element to first.
    # c_array[m-1] is processed first, with carry = initial_carry_for_scan (a_m = 0).
    # This computes a_{m-1} = c_array[m-1] + scan_factor * 0.
    # Then c_array[m-2] is processed, with carry = a_{m-1}.
    # This computes a_{m-2} = c_array[m-2] + scan_factor * a_{m-1}.
    # ...
    # The final carry from scan will be a_0.
    # collected_advantages will be an array [a_0, a_1, ..., a_{m-1}].
    _, collected_advantages = jax.lax.scan(
        scan_body,
        initial_carry_for_scan,
        c_array,  # Input sequence for the scan
        reverse=True,
    )

    return collected_advantages
