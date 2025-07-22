from functools import partial

import jax
import pytest
from flax import struct
from jax import test_util

from hcnn.system_dynamics import SystemDynamicsBase


@struct.dataclass
class SystemDynamicsMock(SystemDynamicsBase):
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        return state


class TestSystemDynamicsBase:
    """Test class for SystemDynamicsBase."""

    state_dim: int = 5
    action_dim: int = 3

    @pytest.fixture
    def sys_dynamics(self) -> SystemDynamicsBase:
        return SystemDynamicsMock(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )

    def test_instance_creation_base(self, sys_dynamics: SystemDynamicsBase):
        assert isinstance(sys_dynamics, SystemDynamicsBase)
        assert isinstance(sys_dynamics.state_dim, int)
        assert isinstance(sys_dynamics.action_dim, int)
        assert sys_dynamics.state_dim == self.state_dim
        assert sys_dynamics.action_dim == self.action_dim

    def test_call_single(self, sys_dynamics: SystemDynamicsBase):
        key = jax.random.key(0)
        state = jax.random.normal(key, (sys_dynamics.state_dim,))
        action = jax.random.normal(key, (sys_dynamics.action_dim,))

        next_state = sys_dynamics(state, action)
        assert next_state.shape == (sys_dynamics.state_dim,)

    def test_call_batch(self, sys_dynamics: SystemDynamicsBase):
        batch_size = 200
        horizon = 20
        key = jax.random.key(0)
        states = jax.random.normal(key, (batch_size, horizon, self.state_dim))
        actions = jax.random.normal(key, (batch_size, horizon, self.action_dim))

        next_states = sys_dynamics(states, actions)
        assert next_states.shape == (
            batch_size,
            horizon,
            sys_dynamics.state_dim,
        )

    def test_call_jit(self, sys_dynamics: SystemDynamicsBase):
        key = jax.random.key(0)
        state = jax.random.normal(key, (self.state_dim,))
        action = jax.random.normal(key, (self.action_dim,))

        jitted_call = jax.jit(lambda s, a: sys_dynamics(s, a))
        next_state = jitted_call(state, action)
        assert next_state.shape == (self.state_dim,)

    def test_call_vjp(self, sys_dynamics: SystemDynamicsBase):
        key = jax.random.key(0)
        state = jax.random.normal(key, (self.state_dim,))
        action = jax.random.normal(key, (self.action_dim,))

        test_util.check_vjp(
            f=sys_dynamics,
            f_vjp=partial(jax.vjp, sys_dynamics),
            args=(state, action),
            rtol=1 / 100,
        )

    def test_call_jvp(self, sys_dynamics: SystemDynamicsBase):
        key = jax.random.key(0)
        state = jax.random.normal(key, (self.state_dim,))
        action = jax.random.normal(key, (self.action_dim,))

        test_util.check_jvp(
            f=sys_dynamics,
            f_jvp=partial(jax.jvp, sys_dynamics),
            args=(state, action),
            rtol=1 / 100,
        )
