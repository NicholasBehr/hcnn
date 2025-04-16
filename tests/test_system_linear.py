import jax.numpy as jnp
import pytest

from hcnn.system.system_linear import SystemLinear


def test_state_matrix_not_2d():
    state_matrix = jnp.array([1])
    input_matrix = jnp.array([[1]])
    with pytest.raises(ValueError, match="Inout and state matrices must be two dimensional!"):
        SystemLinear(state_matrix, input_matrix)


def test_input_matrix_not_2d():
    state_matrix = jnp.array([[1]])
    input_matrix = jnp.array([1])
    with pytest.raises(ValueError, match="Inout and state matrices must be two dimensional!"):
        SystemLinear(state_matrix, input_matrix)


def test_state_matrix_not_square():
    state_matrix = jnp.array([[1, 0]])
    input_matrix = jnp.array([[1]])
    with pytest.raises(ValueError, match="Input matrix must be square!"):
        SystemLinear(state_matrix, input_matrix)


def test_state_and_input_dim_mismatch():
    state_matrix = jnp.array([[1, 0], [0, 1]])
    input_matrix = jnp.array([[1]])
    with pytest.raises(ValueError, match="State dimensions must agree!"):
        SystemLinear(state_matrix, input_matrix)


def test_step_valid():
    state_matrix = jnp.array([[1, 2], [3, 4]])
    input_matrix = jnp.array([[5], [6]])

    system = SystemLinear(state_matrix, input_matrix)
    state = jnp.array([7, 8])
    action = jnp.array([9])

    assert jnp.allclose(jnp.array([68, 107]), system.step(state, action))


STATE_DIM = 2
INPUT_DIM = 1


def test_state_and_input_dim_properties():
    state_matrix = jnp.array([[1, 2], [3, 4]])
    input_matrix = jnp.array([[5], [6]])

    system = SystemLinear(state_matrix, input_matrix)
    assert system.state_dim == STATE_DIM
    assert system.input_dim == INPUT_DIM
