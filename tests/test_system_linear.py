import jax.numpy as jnp
import pytest

from hcnn.system_dynamics.system_dynamics_linear import SystemDynamicsLinear


def test_valid_config_creation():
    state_matrix = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    input_matrix = jnp.array([[5.0], [6.0]])
    config = SystemDynamicsLinear.factory(state_matrix, input_matrix)

    assert config.state_matrix.shape == (2, 2)
    assert config.input_matrix.shape == (2, 1)
    assert config.state_dim == 2  # noqa: PLR2004
    assert config.action_dim == 1


def test_state_matrix_not_square():
    state_matrix = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3, not square
    input_matrix = jnp.array([[1.0], [2.0]])
    with pytest.raises(ValueError, match="Input matrix must be square"):
        SystemDynamicsLinear.factory(state_matrix, input_matrix)


def test_dimension_mismatch_between_state_and_input():
    state_matrix = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
    input_matrix = jnp.array([[1.0], [2.0], [3.0]])  # shape (3, 1)
    with pytest.raises(ValueError, match="State dimensions must agree"):
        SystemDynamicsLinear.factory(state_matrix, input_matrix)


def test_step_basic_case():
    state_matrix = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    input_matrix = jnp.array([[1.0], [1.0]])
    config = SystemDynamicsLinear.factory(state_matrix, input_matrix)

    state = jnp.array([1.0, 2.0])  # shape (2,)
    action = jnp.array([3.0])  # shape (1,)

    next_state = SystemDynamicsLinear.step(config, state, action)

    expected = jnp.array([1.0 + 3.0, 2.0 + 3.0])  # [4.0, 5.0]
    assert jnp.allclose(next_state, expected)
