import jax
import jax.numpy as jnp
import pytest
from flax import struct

from hcnn.system_constraints import SystemConstraintsBase


@struct.dataclass
class SystemConstraintsMock(SystemConstraintsBase):
    @classmethod
    def factory(
        cls,
        state_dim: int,
        action_dim: int,
    ) -> "SystemConstraintsMock":
        return cls(
            state_dim=state_dim,
            action_dim=action_dim,
            constraint_dim=state_dim + action_dim,
            lower_bound=jnp.zeros(state_dim + action_dim),
            upper_bound=jnp.ones(state_dim + action_dim),
        )

    @jax.jit
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        return jnp.concatenate([state, action], axis=-1)


class TestSystemConstraintsBase:
    """Test class for SystemConstraintsBase."""

    @pytest.fixture
    def sys_constraints(self, state_dim, action_dim) -> SystemConstraintsBase:
        return SystemConstraintsMock.factory(
            state_dim=state_dim,
            action_dim=action_dim,
        )

    @pytest.fixture(autouse=True)
    def setup(self, sys_constraints: SystemConstraintsBase):
        self.sys_constraints = sys_constraints

        if self.sys_constraints.state_dim is None:
            self.state_dim = 0
        else:
            self.state_dim = self.sys_constraints.state_dim

        if self.sys_constraints.action_dim is None:
            self.action_dim = 0
        else:
            self.action_dim = self.sys_constraints.action_dim

    def test_instance_creation(self):
        assert isinstance(self.sys_constraints, SystemConstraintsBase)
        assert isinstance(self.sys_constraints.state_dim, int | type(None))
        assert isinstance(self.sys_constraints.action_dim, int | type(None))
        assert isinstance(self.sys_constraints.constraint_dim, int)
        assert isinstance(self.sys_constraints.lower_bound, jax.Array)
        assert isinstance(self.sys_constraints.upper_bound, jax.Array)

    def test_lower_bound(self):
        # 1d float array
        assert jnp.issubdtype(
            self.sys_constraints.lower_bound.dtype, jnp.floating
        )
        assert self.sys_constraints.lower_bound.shape == (
            self.sys_constraints.constraint_dim,
        )

    def test_upper_bound(self):
        # 1d float array
        assert jnp.issubdtype(
            self.sys_constraints.upper_bound.dtype, jnp.floating
        )
        assert self.sys_constraints.upper_bound.shape == (
            self.sys_constraints.constraint_dim,
        )

    def test_call_single(self):
        key = jax.random.key(0)

        # Generate a random state and action
        state = jax.random.normal(key, (self.state_dim,))
        action = jax.random.normal(key, (self.action_dim,))

        constraint_eval = self.sys_constraints(state, action)

        assert constraint_eval.shape == (self.sys_constraints.constraint_dim,)

    def test_call_batch(self):
        batch_size = 200
        horizon = 20
        key = jax.random.key(0)

        # Generate a random state and action
        states = jax.random.normal(key, (batch_size, horizon, self.state_dim))
        actions = jax.random.normal(key, (batch_size, horizon, self.action_dim))

        constraint_eval = self.sys_constraints(states, actions)

        assert constraint_eval.shape == (
            batch_size,
            horizon,
            self.sys_constraints.constraint_dim,
        )
