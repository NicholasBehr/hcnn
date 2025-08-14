import jax.numpy as jnp
import pytest

from hcnn.system_constraints import SystemConstraintsBall

from .test_system_constraints_base import TestSystemConstraintsBase


class TestSystemConstraintsBall(TestSystemConstraintsBase):
    """Test class for SystemConstraintsBall."""

    state_dim: int = 4
    action_dim: int = 3

    @pytest.fixture
    def sys_constraints(self) -> SystemConstraintsBall:
        center = jnp.zeros((self.state_dim,))
        radius = 2.0
        return SystemConstraintsBall.factory(
            center=center, radius=radius, action_dim=self.action_dim
        )

    def test_instance_creation_ball(
        self, sys_constraints: SystemConstraintsBall
    ):
        assert isinstance(sys_constraints, SystemConstraintsBall)
        assert sys_constraints.state_dim == self.state_dim
        assert sys_constraints.action_dim == self.action_dim
        assert sys_constraints.constraint_dim == 1
        assert jnp.all(jnp.isfinite(sys_constraints.lower_bound))
        assert sys_constraints.lower_bound.shape == (1,)
        assert sys_constraints.upper_bound.shape == (1,)
        assert sys_constraints.upper_bound[0] == jnp.inf

    def test_value_signs(self, sys_constraints: SystemConstraintsBall):
        # On the boundary -> 0
        state_boundary = sys_constraints.center.at[0].set(
            sys_constraints.radius
        )
        action = jnp.zeros((self.action_dim,))
        val = sys_constraints(state_boundary, action)
        assert val.shape == (1,)
        assert jnp.allclose(val, jnp.zeros((1,)))

        # Inside -> negative
        state_inside = jnp.zeros((self.state_dim,))
        val_inside = sys_constraints(state_inside, action)
        assert (val_inside < 0).all()

        # Outside -> positive
        state_outside = jnp.ones((self.state_dim,)) * (
            float(sys_constraints.radius) + 0.5
        )
        val_outside = sys_constraints(state_outside, action)
        assert (val_outside > 0).all()
