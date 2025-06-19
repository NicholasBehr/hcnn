import jax.numpy as jnp
import pytest

from hcnn.projection import ProjectionNoSubstitutionNoTerminal
from tests.projection.projection_base import *


@pytest.fixture
def instance(
    policy,
    sys_dynamics,
    sys_constraints,
    horizon: int = 50,
    relaxation_penalty: float = 1.0,
) -> ProjectionNoSubstitutionNoTerminal:
    return ProjectionNoSubstitutionNoTerminal.factory(
        policy=policy,
        sys_dynamics=sys_dynamics,
        sys_constraints=sys_constraints,
        horizon=horizon,
        relaxation_penalty=relaxation_penalty,
    )


def test_get_x0(instance, phi, policy, sys_dynamics, sys_constraints):
    x = instance.get_x0(phi)
    rollout = instance.x_to_rollout(x)

    # (1)
    # !! assumption: no interventions, all contributed to relaxations for x0 !!
    assert jnp.allclose(rollout.interventions, 0)

    # (2)
    # ensure that the rollout satifies the constraints
    # initial state
    assert jnp.allclose(rollout.states[0], phi.initial_state)
    # conform to dynamics
    assert jnp.allclose(
        rollout.states[1:],
        sys_dynamics(rollout.states[:-1], rollout.actions[:-1]),
    )
    # conform to policy
    assert jnp.allclose(
        rollout.actions,
        policy(rollout.states, jax_key=phi.jax_key) + rollout.interventions,
    )
    # state and action constraints
    constr_relax = (
        sys_constraints(rollout.states, rollout.actions)
        + rollout.relaxations_plus[:, None]
        - rollout.relaxations_minus[:, None]
    )
    assert jnp.all(sys_constraints.lower_bound[None, :] <= constr_relax)
    assert jnp.all(constr_relax <= sys_constraints.upper_bound[None, :])
    # ensure positive relaxations
    assert jnp.all(rollout.relaxations_plus >= 0)
    assert jnp.all(rollout.relaxations_minus >= 0)


def test_constraint_fn(instance, phi, sys_constraints):
    x = instance.get_x0(phi)
    constraints = instance.constraint_fn(x, phi)

    desired_shape = (
        instance.state_dim
        + (instance.horizon - 1) * instance.state_dim
        + instance.horizon * instance.action_dim
        + instance.horizon * sys_constraints.constraint_dim
        + 2 * instance.horizon
    )
    assert constraints.shape == (desired_shape,)
