import jax
import jax.numpy as jnp

from hcnn.training import generalized_advantage


def test_td_residuals_single():
    # Define test inputs
    rewards = jnp.array([1.0, 0.5, 0.0])
    values = jnp.array([0.8, 0.6, 0.4])
    td_gamma = 0.9

    # Compute TD residuals
    residuals = generalized_advantage.td_residuals(rewards, values, td_gamma)

    # Define expected output
    expected = jnp.array([0.74, 0.26])

    # Assert equality
    assert jnp.allclose(residuals, expected), (
        f"Expected {expected}, but got {residuals}"
    )


def test_td_residuals_batch():
    # Define test inputs
    rewards = jnp.array([[1.0, 0.5, 0.0], [0.3, 0.2, 4.0]])
    values = jnp.array([[0.8, 0.6, 0.4], [0.5, -3.0, 0.2]])
    td_gamma = 0.9

    # Compute TD residuals
    residuals = generalized_advantage.td_residuals(rewards, values, td_gamma)

    # Define expected output
    expected = jnp.array([[0.74, 0.26], [-2.9, 3.38]])

    # Assert equality
    assert jnp.allclose(residuals, expected), (
        f"Expected {expected}, but got {residuals}"
    )


def test_generalized_advantage_1_single():
    # Define test inputs
    rewards = jnp.array([1.0, 0.5, 0.3, 0.2])
    values = jnp.array([0.8, 0.6, 0.4, 0.7])
    td_gamma = 0.7
    gae_lambda = 0.8

    # Compute GAE
    residuals = generalized_advantage.td_residuals(rewards, values, td_gamma)
    advantages = generalized_advantage.generalized_advantage_1(
        residuals, td_gamma, gae_lambda
    )

    # Define expected output
    advantage_1 = rewards[0] + td_gamma * values[1] - values[0]
    advantage_2 = (
        rewards[0] + td_gamma * rewards[1] + td_gamma**2 * values[2] - values[0]
    )
    advantage_3 = (
        rewards[0]
        + td_gamma * rewards[1]
        + td_gamma**2 * rewards[2]
        + td_gamma**3 * values[3]
        - values[0]
    )
    expected = (1.0 - gae_lambda) * (
        advantage_1 + gae_lambda * advantage_2 + gae_lambda**2 * advantage_3
    )

    # Assert equality
    assert jnp.allclose(advantages, expected), (
        f"Expected {expected}, but got {advantages}"
    )


def test_generalized_advantage_1_batch():
    # Define test inputs for a batch of size 2
    rewards = jnp.array([[1.0, 0.5, 0.3, 0.2], [0.1, 0.9, 0.4, 0.6]])
    values = jnp.array([[0.8, 0.6, 0.4, 0.7], [0.5, 0.2, 0.8, 0.3]])
    td_gamma = 0.7
    gae_lambda = 0.8

    # Compute GAE for the batch
    residuals = generalized_advantage.td_residuals(rewards, values, td_gamma)
    advantages = generalized_advantage.generalized_advantage_1(
        residuals, td_gamma, gae_lambda
    )

    # Define expected output by computing for each item individually
    # and then stacking them. This verifies the batching implementation.
    expected_advantages = []
    for i in range(rewards.shape[0]):
        residuals_i = generalized_advantage.td_residuals(
            rewards[i], values[i], td_gamma
        )
        advantage_i = generalized_advantage.generalized_advantage_1(
            residuals_i, td_gamma, gae_lambda
        )
        expected_advantages.append(advantage_i)

    expected = jnp.array(expected_advantages)

    # Assert equality
    assert advantages.shape == (2,)
    assert jnp.allclose(advantages, expected), (
        f"Expected {expected}, but got {advantages}"
    )


def test_generalized_advantage_t_single():
    # Define test inputs
    keys = jax.random.split(jax.random.key(0), 2)
    trajectory_length = 20
    rewards = jax.random.uniform(
        keys[0], (trajectory_length,), minval=0.0, maxval=1.0
    )
    values = jax.random.uniform(
        keys[1], (trajectory_length,), minval=0.0, maxval=1.0
    )
    td_gamma = 0.7
    gae_lambda = 0.8

    # Compute GAE for all starts
    residuals = generalized_advantage.td_residuals(rewards, values, td_gamma)
    advantages = generalized_advantage.generalized_advantage_t(
        residuals, td_gamma, gae_lambda
    )

    # Define expected output
    advantages_ref = []
    for i in range(len(residuals)):
        # Compute GAE for each starting point
        residuals_i = residuals[i:]
        advantages_i = generalized_advantage.generalized_advantage_1(
            residuals_i, td_gamma, gae_lambda
        )
        advantages_ref.append(advantages_i)

    expected = jnp.array(advantages_ref)

    # Assert equality
    assert advantages.shape == (trajectory_length - 1,)
    assert jnp.allclose(advantages, expected), (
        f"Expected {expected}, but got {advantages}"
    )


def test_generalized_advantage_t_batch():
    # Define test inputs
    keys = jax.random.split(jax.random.key(0), 2)
    batch_size = 5
    trajectory_length = 20
    rewards = jax.random.uniform(
        keys[0], (batch_size, trajectory_length), minval=0.0, maxval=1.0
    )
    values = jax.random.uniform(
        keys[1], (batch_size, trajectory_length), minval=0.0, maxval=1.0
    )
    td_gamma = 0.7
    gae_lambda = 0.8

    # Compute GAE for all starts for the batch
    residuals = generalized_advantage.td_residuals(rewards, values, td_gamma)
    advantages = generalized_advantage.generalized_advantage_t(
        residuals, td_gamma, gae_lambda
    )

    # Define expected output by computing for each item individually
    # and then stacking them. This verifies the batching implementation.
    expected_advantages = []
    for i in range(batch_size):
        # Compute GAE for each trajectory in the batch
        residuals_i = generalized_advantage.td_residuals(
            rewards[i], values[i], td_gamma
        )
        advantages_i = generalized_advantage.generalized_advantage_t(
            residuals_i, td_gamma, gae_lambda
        )
        expected_advantages.append(advantages_i)

    expected = jnp.array(expected_advantages)

    # Assert equality
    assert advantages.shape == (batch_size, trajectory_length - 1)
    assert jnp.allclose(advantages, expected), (
        f"Expected {expected}, but got {advantages}"
    )
