import jax
import jax.numpy as jnp
import pytest

from hcnn.system_cost import SystemCostQuadratic
from tests.system_cost.system_cost_base import *


@pytest.fixture
def instance() -> SystemCostQuadratic:
    key = jax.random.key(0)
    state_cost_root = jax.random.normal(key, (4, 4))
    action_cost_root = jax.random.normal(key, (2, 2))
    return SystemCostQuadratic.factory(
        state_cost=state_cost_root.T @ state_cost_root,
        action_cost=action_cost_root.T @ action_cost_root,
    )


def test_call_example_single():
    instance = SystemCostQuadratic.factory(
        state_cost=jnp.array([[1.0, 2.0], [2.0, 3.0]]),
        action_cost=jnp.array([4.0]),
    )
    state = jnp.array([1.0, 2.0])
    action = jnp.array([3.0])

    cost = instance(state, action)

    expected = jnp.array([57.0])
    assert jnp.allclose(cost, expected)


def test_call_example_batch():
    instance = SystemCostQuadratic.factory(
        state_cost=jnp.array([[1.0, 2.0], [2.0, 3.0]]),
        action_cost=jnp.array([4.0]),
    )
    state = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    action = jnp.array([[3.0], [4.0]])

    cost = instance(state, action)

    expected = jnp.array([57.0, 169.0])
    assert jnp.allclose(cost, expected)
