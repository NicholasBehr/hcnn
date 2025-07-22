from functools import partial

import jax
import jax.numpy as jnp
import pytest
from flax import struct
from jax import test_util

from hcnn.system_costs import SystemCostsBase


@struct.dataclass
class SystemCostMock(SystemCostsBase):
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        return jnp.sum(state, axis=-1) + jnp.sum(action, axis=-1)


class TestSystemCostBase:
    """Test class for SystemCostBase."""

    state_dim: int = 10
    action_dim: int = 5

    @pytest.fixture
    def sys_cost(self) -> SystemCostsBase:
        return SystemCostMock(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )

    def test_instance_creation_base(self, sys_cost: SystemCostsBase):
        assert isinstance(sys_cost, SystemCostsBase)
        assert isinstance(sys_cost.state_dim, int)
        assert isinstance(sys_cost.action_dim, int)
        assert sys_cost.state_dim == self.state_dim
        assert sys_cost.action_dim == self.action_dim

    def test_call_single(self, sys_cost: SystemCostsBase):
        key = jax.random.key(0)
        state = jax.random.normal(key, (sys_cost.state_dim,))
        action = jax.random.normal(key, (sys_cost.action_dim,))

        cost = sys_cost(state, action)
        assert cost.shape == ()

    def test_call_batch(self, sys_cost: SystemCostsBase):
        batch_size = 200
        horizon = 20
        key = jax.random.key(0)
        states = jax.random.normal(
            key, (batch_size, horizon, sys_cost.state_dim)
        )
        actions = jax.random.normal(
            key, (batch_size, horizon, sys_cost.action_dim)
        )

        costs = sys_cost(states, actions)
        assert costs.shape == (
            batch_size,
            horizon,
        )

    def test_call_jit(self, sys_cost: SystemCostsBase):
        key = jax.random.key(0)
        state = jax.random.normal(key, (self.state_dim,))
        action = jax.random.normal(key, (self.action_dim,))

        jitted_call = jax.jit(lambda s, a: sys_cost(s, a))
        next_state = jitted_call(state, action)
        assert next_state.shape == ()

    def test_call_vjp(self, sys_cost: SystemCostsBase):
        key = jax.random.key(0)
        state = jax.random.normal(key, (self.state_dim,))
        action = jax.random.normal(key, (self.action_dim,))

        test_util.check_vjp(
            f=sys_cost,
            f_vjp=partial(jax.vjp, sys_cost),
            args=(state, action),
            rtol=1 / 100,
        )
