from functools import partial

import jax
import jax.numpy as jnp
import pytest
from flax import struct
from jax import test_util

from hcnn.system_constraints import SystemConstraintsBase


@struct.dataclass
class SystemConstraintsMock(SystemConstraintsBase):
    lower_bound: jax.Array
    upper_bound: jax.Array

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

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        return jnp.concatenate([state, action], axis=-1)


class TestSystemConstraintsBase:
    """Test class for SystemConstraintsBase."""

    state_dim: int = 10
    action_dim: int = 5

    @pytest.fixture
    def sys_constraints(self) -> SystemConstraintsBase:
        return SystemConstraintsMock.factory(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )

    def test_instance_creation_base(
        self, sys_constraints: SystemConstraintsBase
    ):
        assert isinstance(sys_constraints, SystemConstraintsBase)
        assert isinstance(sys_constraints.state_dim, int | type(None))
        assert isinstance(sys_constraints.action_dim, int | type(None))
        assert isinstance(sys_constraints.constraint_dim, int)
        assert isinstance(sys_constraints.lower_bound, jax.Array)
        assert isinstance(sys_constraints.upper_bound, jax.Array)

    def test_lower_bound(self, sys_constraints: SystemConstraintsBase):
        # 1d float array
        assert jnp.issubdtype(sys_constraints.lower_bound.dtype, jnp.floating)
        assert sys_constraints.lower_bound.shape == (
            sys_constraints.constraint_dim,
        )

    def test_upper_bound(self, sys_constraints: SystemConstraintsBase):
        # 1d float array
        assert jnp.issubdtype(sys_constraints.upper_bound.dtype, jnp.floating)
        assert sys_constraints.upper_bound.shape == (
            sys_constraints.constraint_dim,
        )

    def test_call_single(self, sys_constraints: SystemConstraintsBase):
        key = jax.random.key(0)

        # Generate a random state and action
        state = jax.random.normal(key, (self.state_dim,))
        action = jax.random.normal(key, (self.action_dim,))

        constraint_eval = sys_constraints(state, action)

        assert constraint_eval.shape == (sys_constraints.constraint_dim,)

    def test_call_batch(self, sys_constraints: SystemConstraintsBase):
        batch_size = 200
        horizon = 20
        key = jax.random.key(0)

        # Generate a random state and action
        states = jax.random.normal(key, (batch_size, horizon, self.state_dim))
        actions = jax.random.normal(key, (batch_size, horizon, self.action_dim))

        constraint_eval = sys_constraints(states, actions)

        assert constraint_eval.shape == (
            batch_size,
            horizon,
            sys_constraints.constraint_dim,
        )

    def test_call_jit(self, sys_constraints: SystemConstraintsBase):
        key = jax.random.key(0)
        state = jax.random.normal(key, (self.state_dim,))
        action = jax.random.normal(key, (self.action_dim,))

        jitted_call = jax.jit(lambda s, a: sys_constraints(s, a))
        next_state = jitted_call(state, action)
        assert next_state.shape == (sys_constraints.constraint_dim,)

    def test_call_vjp(self, sys_constraints: SystemConstraintsBase):
        key = jax.random.key(0)
        state = jax.random.normal(key, (self.state_dim,))
        action = jax.random.normal(key, (self.action_dim,))

        test_util.check_vjp(
            f=sys_constraints,
            f_vjp=partial(jax.vjp, sys_constraints),
            args=(state, action),
            rtol=1 / 100,
        )
