import jax
import jax.numpy as jnp
import pytest

from hcnn.system_constraints import SystemConstraintsNone

from .test_system_constraints_base import TestSystemConstraintsBase


class TestSystemConstraintsNone(TestSystemConstraintsBase):
    """Test class for SystemConstraintsNone."""

    state_dim: int = 6
    action_dim: int = 2

    @pytest.fixture
    def sys_constraints(self) -> SystemConstraintsNone:
        return SystemConstraintsNone.factory(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )

    def test_instance_creation_none(
        self, sys_constraints: SystemConstraintsNone
    ):
        """Test that SystemConstraintsNone instance is created correctly."""
        assert isinstance(sys_constraints, SystemConstraintsNone)
        assert sys_constraints.state_dim == self.state_dim
        assert sys_constraints.action_dim == self.action_dim
        assert sys_constraints.constraint_dim == 1
        assert sys_constraints.lower_bound.shape == (1,)
        assert sys_constraints.upper_bound.shape == (1,)
        assert jnp.issubdtype(sys_constraints.lower_bound.dtype, jnp.floating)
        assert jnp.issubdtype(sys_constraints.upper_bound.dtype, jnp.floating)

    def test_call_returns_empty_constraints(
        self, sys_constraints: SystemConstraintsNone
    ):
        """Test that calling the constraint function returns empty arrays."""
        key = jax.random.key(0)

        # Test single sample
        state = jax.random.normal(key, (self.state_dim,))
        action = jax.random.normal(key, (self.action_dim,))

        constraints = sys_constraints(state, action)
        assert constraints.shape == (1,)

        # Test batch
        batch_size = 10
        states = jax.random.normal(key, (batch_size, self.state_dim))
        actions = jax.random.normal(key, (batch_size, self.action_dim))

        constraints_batch = sys_constraints(states, actions)
        assert constraints_batch.shape == (batch_size, 1)

        # Test batch with horizon
        horizon = 20
        states_horizon = jax.random.normal(
            key, (batch_size, horizon, self.state_dim)
        )
        actions_horizon = jax.random.normal(
            key, (batch_size, horizon, self.action_dim)
        )

        constraints_horizon = sys_constraints(states_horizon, actions_horizon)
        assert constraints_horizon.shape == (batch_size, horizon, 1)

    def test_unconstrained_property(
        self, sys_constraints: SystemConstraintsNone
    ):
        """Test that the system is truly unconstrained."""
        # Since there are no constraints, any state-action pair should be valid
        # This is implicitly tested by having constraint_dim = 0
        assert sys_constraints.constraint_dim == 1
        assert len(sys_constraints.lower_bound) == 1
        assert len(sys_constraints.upper_bound) == 1
