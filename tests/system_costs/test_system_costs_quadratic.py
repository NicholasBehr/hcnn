import jax
import jax.numpy as jnp
import pytest

from hcnn.system_costs import SystemCostsQuadratic

from .test_system_costs_base import TestSystemCostBase


class TestSystemCostsQuadratic(TestSystemCostBase):
    """Test class for SystemCostQuadratic."""

    @pytest.fixture
    def sys_cost(self) -> SystemCostsQuadratic:
        key = jax.random.key(0)
        state_cost_root = jax.random.normal(
            key, (self.state_dim, self.state_dim)
        )
        action_cost_root = jax.random.normal(
            key, (self.action_dim, self.action_dim)
        )
        return SystemCostsQuadratic.factory(
            state_cost_matrix=state_cost_root.T @ state_cost_root,
            action_cost_matrix=action_cost_root.T @ action_cost_root,
        )

    def test_instance_creation_quadratic(self, sys_cost: SystemCostsQuadratic):
        assert isinstance(sys_cost, SystemCostsQuadratic)
        assert sys_cost.state_cost_matrix.shape == (
            self.state_dim,
            self.state_dim,
        )
        assert sys_cost.action_cost_matrix.shape == (
            self.action_dim,
            self.action_dim,
        )

    def test_call_example_single(self, sys_cost: SystemCostsQuadratic):
        instance = SystemCostsQuadratic.factory(
            state_cost_matrix=jnp.array([[1.0, 2.0], [2.0, 3.0]]),
            action_cost_matrix=jnp.array([4.0]),
        )
        state = jnp.array([1.0, 2.0])
        action = jnp.array([3.0])

        cost = instance(state, action)

        expected = jnp.array([57.0])
        assert jnp.allclose(cost, expected)

    def test_call_example_batch(self):
        instance = SystemCostsQuadratic.factory(
            state_cost_matrix=jnp.array([[1.0, 2.0], [2.0, 3.0]]),
            action_cost_matrix=jnp.array([4.0]),
        )
        state = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        action = jnp.array([[3.0], [4.0]])

        cost = instance(state, action)

        expected = jnp.array([57.0, 169.0])
        assert jnp.allclose(cost, expected)
