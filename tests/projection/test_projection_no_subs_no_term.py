import jax.numpy as jnp
import pytest

from hcnn.projection.projection_no_subs_no_term import (
    ProjectionNoSubstitutionNoTerminal,
)
from tests.projection.test_projection_base import TestProjectionBase


class TestProjectionNoSubstitutionNoTerminal(TestProjectionBase):
    """Test class for ProjectionNoSubstitutionNoTerminal."""

    @pytest.fixture
    def projection(
        self,
        policy,
        sys_dynamics,
        sys_constraints,
        horizon,
    ) -> ProjectionNoSubstitutionNoTerminal:
        return ProjectionNoSubstitutionNoTerminal.factory(
            policy=policy,
            sys_dynamics=sys_dynamics,
            sys_constraints=sys_constraints,
            horizon=horizon,
            relaxation_penalty=1.0,
        )

    def test_get_x0(self):
        x = self.projection.get_x0(self.phi)
        rollout = self.projection.x_to_rollout(x)

        # (1)
        # !! assumption: no interventions, all contributed to relaxations for x0 !!
        assert jnp.allclose(rollout.interventions, 0)

        # (2)
        # ensure that the rollout satifies the constraints
        # initial state
        assert jnp.allclose(rollout.states[0], self.phi.initial_state)
        # conform to dynamics
        assert jnp.allclose(
            rollout.states[1:],
            self.sys_dynamics(rollout.states[:-1], rollout.actions[:-1]),
        )
        # conform to policy
        assert jnp.allclose(
            rollout.actions,
            self.policy(rollout.states, jax_key=self.phi.jax_key)
            + rollout.interventions,
        )
        # state and action constraints
        constr_relax = (
            self.sys_constraints(rollout.states, rollout.actions)
            + rollout.relaxations_plus
            - rollout.relaxations_minus
        )
        assert jnp.all(
            self.sys_constraints.lower_bound[None, :] <= constr_relax
        )
        assert jnp.all(
            constr_relax <= self.sys_constraints.upper_bound[None, :]
        )
        # ensure positive relaxations
        assert jnp.all(rollout.relaxations_plus >= 0)
        assert jnp.all(rollout.relaxations_minus >= 0)

    def test_constraint_fn(self):
        x = self.projection.get_x0(self.phi)
        constraints = self.projection.constraint_fn(x, self.phi)

        desired_shape = (
            self.state_dim
            + (self.horizon - 1) * self.state_dim
            + self.horizon * self.action_dim
            + self.horizon * self.constraint_dim
            + 2 * self.horizon * self.constraint_dim
        )
        assert constraints.shape == (desired_shape,)
