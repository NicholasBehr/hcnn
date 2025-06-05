import jax
import jax.numpy as jnp

from hcnn.training import generalized_advantage


def test_td_residuals():
    # Define test inputs
    rewards = jnp.array([1.0, 0.5, 0.0])
    values = jnp.array([0.8, 0.6, 0.4])
    td_gamma = 0.9

    # Compute TD residuals
    residuals = generalized_advantage.td_residuals(rewards, values, td_gamma)

    # Define expected output
    expected = jnp.array([0.74, 0.26])

    # Assert equality
    assert jnp.allclose(residuals, expected), f"Expected {expected}, but got {residuals}"


def test_generalized_advantage_1():
    # Define test inputs
    rewards = jnp.array([1.0, 0.5, 0.3, 0.2])
    values = jnp.array([0.8, 0.6, 0.4, 0.7])
    td_gamma = 0.7
    gae_lambda = 0.8

    # Compute GAE
    residuals = generalized_advantage.td_residuals(rewards, values, td_gamma)
    advantages = generalized_advantage.generalized_advantage_1(residuals, td_gamma, gae_lambda)

    # Define expected output
    advantage_1 = rewards[0] + td_gamma * values[1] - values[0]
    advantage_2 = rewards[0] + td_gamma * rewards[1] + td_gamma**2 * values[2] - values[0]
    advantage_3 = rewards[0] + td_gamma * rewards[1] + td_gamma**2 * rewards[2] + td_gamma**3 * values[3] - values[0]
    expected = (1.0 - gae_lambda) * (advantage_1 + gae_lambda * advantage_2 + gae_lambda**2 * advantage_3)

    # Assert equality
    assert jnp.allclose(advantages, expected), f"Expected {expected}, but got {advantages}"


def test_generalized_advantage_t():
    # Define test inputs
    keys = jax.random.split(jax.random.key(0), 2)
    rewards = jax.random.uniform(keys[0], (20,), minval=0.0, maxval=1.0)
    values = jax.random.uniform(keys[1], (20,), minval=0.0, maxval=1.0)
    td_gamma = 0.7
    gae_lambda = 0.8

    # Compute GAE for all starts
    residuals = generalized_advantage.td_residuals(rewards, values, td_gamma)
    advantages = generalized_advantage.generalized_advantage_t(residuals, td_gamma, gae_lambda)

    # Define expected output
    advantages_ref = []
    for i in range(len(residuals)):
        # Compute GAE for each starting point
        residuals_i = residuals[i:]
        advantages_i = generalized_advantage.generalized_advantage_1(residuals_i, td_gamma, gae_lambda)
        advantages_ref.append(advantages_i)

    expected = jnp.array(advantages_ref)

    # Assert equality
    assert jnp.allclose(advantages, expected), f"Expected {expected}, but got {advantages}"
