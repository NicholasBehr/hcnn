import jax.numpy as jnp
import pytest

from hcnn.system.system_linear import SystemLinear


def test_matrix_a_is_not_2d():
    matrix_a = jnp.array([1])
    matrix_b = jnp.array([[1]])
    with pytest.raises(ValueError):
        SystemLinear(matrix_a, matrix_b)


def test_matrix_b_is_not_2d():
    matrix_a = jnp.array([[1]])
    matrix_b = jnp.array([1])
    with pytest.raises(ValueError):
        SystemLinear(matrix_a, matrix_b)


def test_matrix_a_is_not_square():
    matrix_a = jnp.array([[1, 0]])
    matrix_b = jnp.array([[1]])
    with pytest.raises(ValueError):
        SystemLinear(matrix_a, matrix_b)


def test_state_dim_disagree():
    matrix_a = jnp.array([[1, 0], [0, 1]])
    matrix_b = jnp.array([[1]])
    with pytest.raises(ValueError):
        SystemLinear(matrix_a, matrix_b)


def test_step_working():
    matrix_a = jnp.array([[1, 2], [3, 4]])
    matrix_b = jnp.array([[5], [6]])

    system = SystemLinear(matrix_a, matrix_b)
    state = jnp.array([7, 8])
    action = jnp.array([9])

    assert jnp.allclose(jnp.array([68, 107]), system.step(state, action))
