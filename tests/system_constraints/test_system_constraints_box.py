import jax.numpy as jnp
import pytest

from hcnn.system_constraints.system_constraints_box import SystemConstraintsBox
from tests.system_constraints.test_system_constraints_base import (
    TestSystemConstraintsBase,
)


class TestSystemConstraintsBox(TestSystemConstraintsBase):
    """Test class for SystemConstraintsBox."""

    @pytest.fixture
    def sys_constraints(self) -> SystemConstraintsBox:
        return SystemConstraintsBox.factory(
            state_l=jnp.array([1.0, -jnp.inf, 1.0, -jnp.inf, -2.0, -3.0]),
            state_u=jnp.array([3.0, 1.0, jnp.inf, jnp.inf, -1.0, 4.0]),
            action_l=jnp.array([0.0, -jnp.inf]),
            action_u=jnp.array([1.0, jnp.inf]),
        )

    @pytest.fixture(autouse=True)
    def setup(
        self,
        sys_constraints: SystemConstraintsBox,
    ):
        self.sys_constraints = sys_constraints
        self.state_dim = 6
        self.action_dim = 2

    def test_instance_creation(self):
        assert isinstance(self.sys_constraints, SystemConstraintsBox)
        assert self.sys_constraints.state_dim == 6
        assert self.sys_constraints.action_dim == 2
        assert self.sys_constraints.constraint_dim == 6
        assert jnp.allclose(
            self.sys_constraints.lower_bound,
            jnp.array([1.0, -jnp.inf, 1.0, -2.0, -3.0, 0.0]),
        )
        assert jnp.allclose(
            self.sys_constraints.upper_bound,
            jnp.array([3.0, 1.0, jnp.inf, -1.0, 4.0, 1.0]),
        )
        assert jnp.allclose(
            self.sys_constraints.active_indices, jnp.array([0, 1, 2, 4, 5, 6])
        )
