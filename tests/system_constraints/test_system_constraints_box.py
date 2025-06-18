import jax.numpy as jnp
import pytest

from hcnn.system_constraints import SystemConstraintsBox
from tests.system_constraints.system_constraints_base import *


@pytest.fixture
def instance() -> SystemConstraintsBox:
    return SystemConstraintsBox.factory(
        state_l=jnp.array([1.0, -jnp.inf, 1.0, -jnp.inf, -2.0, -3.0]),
        state_u=jnp.array([3.0, 1.0, jnp.inf, jnp.inf, -1.0, 4.0]),
        action_l=jnp.array([0.0, -jnp.inf]),
        action_u=jnp.array([1.0, jnp.inf]),
    )


def test_valid_instance_creation(instance: SystemConstraintsBox):
    assert instance.state_dim == 6
    assert instance.action_dim == 2
    assert instance.constraint_dim == 6
    assert jnp.allclose(
        instance.lower_bound, jnp.array([1.0, -jnp.inf, 1.0, -2.0, -3.0, 0.0])
    )
    assert jnp.allclose(
        instance.upper_bound, jnp.array([3.0, 1.0, jnp.inf, -1.0, 4.0, 1.0])
    )
    assert jnp.allclose(instance.active_indices, jnp.array([0, 1, 2, 4, 5, 6]))
