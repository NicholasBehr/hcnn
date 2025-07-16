import jax
import pytest

from hcnn.system_constraints import SystemConstraintsSphere

from .test_system_constraints_base import TestSystemConstraintsBase


class TestSystemConstraintsSphere(TestSystemConstraintsBase):
    """Test class for SystemConstraintsSphere."""

    @pytest.fixture
    def sys_constraints(self) -> SystemConstraintsSphere:
        key = jax.random.key(0)
        return SystemConstraintsSphere.factory(
            center=jax.random.normal(key, (self.state_dim,)),
            radius=jax.random.normal(key),
        )

    def test_instance_creation(self, sys_constraints: SystemConstraintsSphere):
        assert isinstance(sys_constraints, SystemConstraintsSphere)
        assert sys_constraints.state_dim == self.state_dim
        assert sys_constraints.action_dim is None
        assert sys_constraints.constraint_dim == 1
        assert sys_constraints.center.shape == (self.state_dim,)
