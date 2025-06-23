import jax
import jax.numpy as jnp
import pytest

from hcnn.system_dynamics import SystemDynamicsLinear

from .test_system_dynamics_base import TestSystemDynamicsBase


class TestSystemDynamicsLinear(TestSystemDynamicsBase):
    @pytest.fixture
    def sys_dynamics(self, state_dim, action_dim) -> SystemDynamicsLinear:
        key = jax.random.key(0)
        return SystemDynamicsLinear.factory(
            state_matrix=jax.random.normal(key, (state_dim, state_dim)),
            input_matrix=jax.random.normal(key, (state_dim, action_dim)),
        )

    @pytest.fixture(autouse=True)
    def setup(
        self,
        sys_dynamics: SystemDynamicsLinear,
        state_dim: int,
        action_dim: int,
    ):
        self.sys_dynamics = sys_dynamics
        self.state_dim = state_dim
        self.action_dim = action_dim

    def test_instance_creation(self):
        assert isinstance(self.sys_dynamics, SystemDynamicsLinear)
        assert self.sys_dynamics.state_dim == self.state_dim
        assert self.sys_dynamics.action_dim == self.action_dim
        assert self.sys_dynamics.state_matrix_transp.shape == (
            self.state_dim,
            self.state_dim,
        )
        assert self.sys_dynamics.input_matrix_transp.shape == (
            self.action_dim,
            self.state_dim,
        )

    def test_state_matrix_not_square(self):
        state_matrix = jnp.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )  # 2x3, not square
        input_matrix = jnp.array([[1.0], [2.0]])
        with pytest.raises(AssertionError):
            SystemDynamicsLinear.factory(state_matrix, input_matrix)

    def test_dimension_mismatch_between_state_and_input(self):
        state_matrix = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
        input_matrix = jnp.array([[1.0], [2.0], [3.0]])  # shape (3, 1)
        with pytest.raises(AssertionError):
            SystemDynamicsLinear.factory(state_matrix, input_matrix)

    def test_call_example_single(self):
        state_matrix = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        input_matrix = jnp.array([[5.0], [6.0]])
        instance = SystemDynamicsLinear.factory(state_matrix, input_matrix)

        state = jnp.array([1.0, 2.0])  # shape (2,)
        action = jnp.array([3.0])  # shape (1,)

        next_state = instance(state, action)

        expected = jnp.array([20.0, 29.0])
        assert jnp.allclose(next_state, expected)

    def test_call_example_batch(self):
        state_matrix = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        input_matrix = jnp.array([[5.0], [6.0]])
        instance = SystemDynamicsLinear.factory(state_matrix, input_matrix)

        states = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # shape (2,2)
        actions = jnp.array([[3.0], [4.0]])  # shape (2,1)

        next_states = instance(states, actions)

        expected = jnp.array([[20.0, 29.0], [31.0, 49.0]])
        assert jnp.allclose(next_states, expected)
