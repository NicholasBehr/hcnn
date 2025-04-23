import jax.numpy as jnp
import pytest

from hcnn.constraint.constraint_state_outside_sphere import ConstraintStateOutsideSphere
from hcnn.policy.policy_lqr import PolicyLQR
from hcnn.projector.projector import Projector
from hcnn.system.system_linear import SystemLinear


@pytest.fixture
def linear_system():
    state_matrix = jnp.array([[1, -1], [1, 0]])
    input_matrix = jnp.array([[1.0], [0.0]])
    return SystemLinear.factory(state_matrix=state_matrix, input_matrix=input_matrix)


@pytest.fixture
def linear_policy(linear_system):
    state_cost = jnp.eye(2)
    input_cost = jnp.eye(1)
    return PolicyLQR.factory(linear_system, state_cost, input_cost)


@pytest.fixture
def constraint_state_outside_sphere():
    center = jnp.array([2.0, 2.0])
    radius = 1
    return ConstraintStateOutsideSphere.factory(center=center, radius=radius)


@pytest.fixture
def projector(linear_system, linear_policy, constraint_state_outside_sphere):
    # Create a projector instance
    return Projector(
        system=linear_system,
        policy=linear_policy,
        constraint=constraint_state_outside_sphere,
        horizon=4,
    )


def test_warmstart(projector):
    horizon = 4

    # Create a dummy optimization variable
    initial_state = jnp.array([2, -2])

    # Perform warmstart
    warmstart_x = projector.warm_start(initial_state)
    actions = warmstart_x.actions
    policy_actions = warmstart_x.policy_actions
    states = warmstart_x.states
    relaxations = warmstart_x.relaxations
    max_relaxation = warmstart_x.max_relaxation

    assert jnp.all(states[0] == initial_state)  # initial state
    for i in range(0, horizon - 2):
        assert jnp.all(states[i + 1] == projector.system.step(states[i], actions[i]))  # dynamics
        assert jnp.all(actions[i] == policy_actions[i])  # policy action
        assert jnp.all(
            projector.constraint.function(states[i], actions[i]) + relaxations[i]
            <= projector.constraint.upper_bound
        )  # constraint upper bound
        assert jnp.all(
            projector.constraint.lower_bound
            <= projector.constraint.function(states[i], actions[i]) + relaxations[i]
        )  # constraint lower bound
        assert max_relaxation == jnp.max(jnp.abs(relaxations))  # relaxation


def test_constraint_initial(projector):
    # Create a dummy optimization variable
    initial_state = jnp.array([2, -2])

    # Perform warmstart
    warmstart_x = projector.warm_start(initial_state)

    assert jnp.allclose(projector.constraint_initial(warmstart_x, initial_state), 0)
    assert not jnp.allclose(
        projector.constraint_initial(warmstart_x, initial_state=jnp.array([2, -1])), 0
    )


def test_constraint_dynamics(projector):
    # Create a dummy optimization variable
    initial_state = jnp.array([2, -2])

    # Perform warmstart
    warmstart_x = projector.warm_start(initial_state)

    assert jnp.allclose(projector.constraint_dynamics(warmstart_x), 0)
    assert not jnp.allclose(
        projector.constraint_dynamics(
            warmstart_x.replace(states=warmstart_x.states.at[2].set(jnp.array([2, -1])))
        ),
        0,
    )


def test_constraint_policy(projector):
    # Create a dummy optimization variable
    initial_state = jnp.array([2, -2])

    # Perform warmstart
    warmstart_x = projector.warm_start(initial_state)

    assert jnp.allclose(
        projector.constraint_policy(warmstart_x),
        0,
        atol=1e-8,
    )
    assert not jnp.allclose(
        projector.constraint_policy(
            warmstart_x.replace(policy_actions=warmstart_x.policy_actions.at[2].set(jnp.array([2])))
        ),
        0,
        atol=1e-8,
    )
