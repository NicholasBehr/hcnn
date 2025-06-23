import jax
import pytest

from hcnn.system_constraints import SystemConstraintsSphere

from .test_system_constraints_base import TestSystemConstraintsBase


class TestSystemConstraintsSphere(TestSystemConstraintsBase):
    """Test class for SystemConstraintsSphere."""

    @pytest.fixture
    def sys_constraints(self, state_dim) -> SystemConstraintsSphere:
        key = jax.random.key(0)
        return SystemConstraintsSphere.factory(
            center=jax.random.normal(key, (state_dim,)),
            radius=jax.random.normal(key),
        )

    @pytest.fixture(autouse=True)
    def setup(
        self,
        sys_constraints: SystemConstraintsSphere,
        state_dim: int,
        action_dim: int,
    ):
        self.sys_constraints = sys_constraints
        self.state_dim = state_dim
        self.action_dim = action_dim

    def test_instance_creation(self):
        assert isinstance(self.sys_constraints, SystemConstraintsSphere)
        assert self.sys_constraints.state_dim == self.state_dim
        assert self.sys_constraints.action_dim is None
        assert self.sys_constraints.constraint_dim == 1
        assert self.sys_constraints.center.shape == (self.state_dim,)
