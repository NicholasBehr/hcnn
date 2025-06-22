import jax
import jax.numpy as jnp
import pytest
from flax import struct

from hcnn.system_costs.system_costs_base import SystemCostsBase


@struct.dataclass
class SystemCostMock(SystemCostsBase):
    @jax.jit
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        return jnp.sum(state, axis=-1) + jnp.sum(action, axis=-1)


class TestSystemCostBase:
    """Test class for SystemCostBase."""

    @pytest.fixture
    def sys_cost(self, state_dim, action_dim) -> SystemCostsBase:
        return SystemCostMock(
            state_dim=state_dim,
            action_dim=action_dim,
        )

    @pytest.fixture(autouse=True)
    def setup(self, sys_cost: SystemCostsBase):
        self.sys_cost = sys_cost
        self.state_dim = sys_cost.state_dim
        self.action_dim = sys_cost.action_dim

    def test_instance_creation(self):
        assert isinstance(self.sys_cost, SystemCostsBase)
        assert isinstance(self.sys_cost.state_dim, int)
        assert isinstance(self.sys_cost.action_dim, int)

    def test_call_single(self):
        key = jax.random.key(0)
        state = jax.random.normal(key, (self.state_dim,))
        action = jax.random.normal(key, (self.action_dim,))

        cost = self.sys_cost(state, action)
        assert cost.shape == ()

    def test_call_batch(self):
        batch_size = 200
        horizon = 20
        key = jax.random.key(0)
        states = jax.random.normal(key, (batch_size, horizon, self.state_dim))
        actions = jax.random.normal(key, (batch_size, horizon, self.action_dim))

        costs = self.sys_cost(states, actions)
        assert costs.shape == (
            batch_size,
            horizon,
        )
