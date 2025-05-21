import jax
import jax.numpy as jnp
import pytest

from hcnn.system_dynamics import SystemDynamicsLinear
from tests.system_dynamics.system_dynamics_base import *


@pytest.fixture
def instance() -> SystemDynamicsLinear:
    key = jax.random.key(0)
    return SystemDynamicsLinear.factory(
        state_matrix=jax.random.normal(key, (4, 4)),
        input_matrix=jax.random.normal(key, (4, 2)),
    )


def test_valid_instance_creation(instance: SystemDynamicsLinear):
    assert instance.state_dim == 4
    assert instance.action_dim == 2
    assert instance.state_matrix.shape == (4, 4)
    assert instance.input_matrix.shape == (4, 2)


def test_state_matrix_not_square():
    state_matrix = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3, not square
    input_matrix = jnp.array([[1.0], [2.0]])
    with pytest.raises(AssertionError):
        SystemDynamicsLinear.factory(state_matrix, input_matrix)


def test_dimension_mismatch_between_state_and_input():
    state_matrix = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
    input_matrix = jnp.array([[1.0], [2.0], [3.0]])  # shape (3, 1)
    with pytest.raises(AssertionError):
        SystemDynamicsLinear.factory(state_matrix, input_matrix)


def test_next_state_example():
    state_matrix = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    input_matrix = jnp.array([[1.0], [1.0]])
    instance = SystemDynamicsLinear.factory(state_matrix, input_matrix)

    state = jnp.array([1.0, 2.0])  # shape (2,)
    action = jnp.array([3.0])  # shape (1,)

    next_state = instance.next_state(state, action)

    expected = jnp.array([1.0 + 3.0, 2.0 + 3.0])  # [4.0, 5.0]
    assert jnp.allclose(next_state, expected)
