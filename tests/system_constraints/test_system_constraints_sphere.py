import jax
import pytest

from hcnn.system_constraints import SystemConstraintsSphere
from tests.system_constraints.system_constraints_base import *


@pytest.fixture
def instance() -> SystemConstraintsSphere:
    key = jax.random.key(0)
    return SystemConstraintsSphere.factory(
        center=jax.random.normal(key, (4,)),
        radius=jax.random.normal(key),
    )


def test_valid_instance_creation(instance: SystemConstraintsSphere):
    assert instance.state_dim == 4
    assert instance.action_dim is None
    assert instance.constraint_dim == 1
    assert instance.lower_bound.shape == (1,)
    assert instance.upper_bound.shape == (1,)
    assert instance.center.shape == (4,)
