import jax
import pytest
from flax import struct

from hcnn.system_dynamics.system_dynamics_base import SystemDynamicsBase


@struct.dataclass
class SystemDynamicsMock(SystemDynamicsBase):
    @jax.jit
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        return state


class TestSystemDynamicsBase:
    """Test class for SystemDynamicsBase."""

    @pytest.fixture
    def sys_dynamics(self, state_dim, action_dim) -> SystemDynamicsBase:
        return SystemDynamicsMock(
            state_dim=state_dim,
            action_dim=action_dim,
        )

    @pytest.fixture(autouse=True)
    def setup(self, sys_dynamics: SystemDynamicsBase):
        self.sys_dynamics = sys_dynamics
        self.state_dim = sys_dynamics.state_dim
        self.action_dim = sys_dynamics.action_dim

    def test_instance_creation(self):
        assert isinstance(self.sys_dynamics, SystemDynamicsBase)
        assert isinstance(self.sys_dynamics.state_dim, int)
        assert isinstance(self.sys_dynamics.action_dim, int)

    def test_call_single(self):
        key = jax.random.key(0)
        state = jax.random.normal(key, (self.state_dim,))
        action = jax.random.normal(key, (self.action_dim,))

        next_state = self.sys_dynamics(state, action)
        assert next_state.shape == (self.state_dim,)

    def test_call_batch(self):
        batch_size = 200
        horizon = 20
        key = jax.random.key(0)
        states = jax.random.normal(key, (batch_size, horizon, self.state_dim))
        actions = jax.random.normal(key, (batch_size, horizon, self.action_dim))

        next_states = self.sys_dynamics(states, actions)
        assert next_states.shape == (
            batch_size,
            horizon,
            self.state_dim,
        )
