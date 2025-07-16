import jax.numpy as jnp
import pytest

from hcnn.system_constraints import SystemConstraintsBox

from .test_system_constraints_base import TestSystemConstraintsBase


class TestSystemConstraintsBox(TestSystemConstraintsBase):
    """Test class for SystemConstraintsBox."""

    state_dim: int = 6
    action_dim: int = 2

    @pytest.fixture
    def sys_constraints(self) -> SystemConstraintsBox:
        return SystemConstraintsBox.factory(
            state_l=jnp.array([1.0, -jnp.inf, 1.0, -jnp.inf, -2.0, -3.0]),
            state_u=jnp.array([3.0, 1.0, jnp.inf, jnp.inf, -1.0, 4.0]),
            action_l=jnp.array([0.0, -jnp.inf]),
            action_u=jnp.array([1.0, jnp.inf]),
        )

    def test_instance_creation_box(self, sys_constraints: SystemConstraintsBox):
        assert isinstance(sys_constraints, SystemConstraintsBox)
        assert sys_constraints.state_dim == 6
        assert sys_constraints.action_dim == 2
        assert sys_constraints.constraint_dim == 6
        assert jnp.allclose(
            sys_constraints.lower_bound,
            jnp.array([1.0, -jnp.inf, 1.0, -2.0, -3.0, 0.0]),
        )
        assert jnp.allclose(
            sys_constraints.upper_bound,
            jnp.array([3.0, 1.0, jnp.inf, -1.0, 4.0, 1.0]),
        )
        assert jnp.allclose(
            sys_constraints.active_indices, jnp.array([0, 1, 2, 4, 5, 6])
        )
