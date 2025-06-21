import jax
import jax.numpy as jnp
import pytest

from hcnn.system_cost.system_cost_quadratic import SystemCostQuadratic
from tests.system_cost.test_system_cost_base import TestSystemCostBase


class TestSystemCostQuadratic(TestSystemCostBase):
    """Test class for SystemCostQuadratic."""

    @pytest.fixture
    def sys_cost(self, state_dim, action_dim) -> SystemCostQuadratic:
        key = jax.random.key(0)
        state_cost_root = jax.random.normal(key, (state_dim, state_dim))
        action_cost_root = jax.random.normal(key, (action_dim, action_dim))
        return SystemCostQuadratic.factory(
            state_cost_matrix=state_cost_root.T @ state_cost_root,
            action_cost_matrix=action_cost_root.T @ action_cost_root,
        )

    @pytest.fixture(autouse=True)
    def setup(
        self,
        sys_cost: SystemCostQuadratic,
        state_dim: int,
        action_dim: int,
    ):
        self.sys_cost = sys_cost
        self.state_dim = state_dim
        self.action_dim = action_dim

    def test_instance_creation(self):
        assert isinstance(self.sys_cost, SystemCostQuadratic)
        assert self.sys_cost.state_cost_matrix.shape == (
            self.state_dim,
            self.state_dim,
        )
        assert self.sys_cost.action_cost_matrix.shape == (
            self.action_dim,
            self.action_dim,
        )

    def test_call_example_single(self):
        instance = SystemCostQuadratic.factory(
            state_cost_matrix=jnp.array([[1.0, 2.0], [2.0, 3.0]]),
            action_cost_matrix=jnp.array([4.0]),
        )
        state = jnp.array([1.0, 2.0])
        action = jnp.array([3.0])

        cost = instance(state, action)

        expected = jnp.array([57.0])
        assert jnp.allclose(cost, expected)

    def test_call_example_batch(self):
        instance = SystemCostQuadratic.factory(
            state_cost_matrix=jnp.array([[1.0, 2.0], [2.0, 3.0]]),
            action_cost_matrix=jnp.array([4.0]),
        )
        state = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        action = jnp.array([[3.0], [4.0]])

        cost = instance(state, action)

        expected = jnp.array([57.0, 169.0])
        assert jnp.allclose(cost, expected)
