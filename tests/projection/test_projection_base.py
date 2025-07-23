from functools import partial

import jax
import jax.numpy as jnp
import pytest
from flax import nnx, struct
from jax import test_util

from hcnn.policy import PolicyBase
from hcnn.projection import (
    ProjectionBase,
    ProjectionPhiD,
    ProjectionPhiND,
    Rollout,
)
from hcnn.system_constraints import SystemConstraintsBase
from hcnn.system_dynamics import SystemDynamicsBase
from hcnn.training.dropout_mlp import DropoutMLP
from tests.policy import PolicyMock
from tests.system_constraints import SystemConstraintsMock
from tests.system_dynamics import SystemDynamicsMock


@struct.dataclass
class ProjectionMock(ProjectionBase):
    @classmethod
    def factory(
        cls,
        policy,
        relax_to_go,
        sys_dynamics,
        sys_constraints,
        horizon,
    ) -> "ProjectionMock":
        state_dim = sys_constraints.state_dim
        action_dim = sys_constraints.action_dim
        constraint_dim = sys_constraints.constraint_dim
        policy_graphdef = nnx.split(policy)[0]
        relax_to_go_graphdef = nnx.split(relax_to_go)[0]
        return cls(
            cost_quad_diag=jnp.zeros(
                (state_dim + 2 * action_dim + 2 * constraint_dim) * horizon
            ),
            cost_linear=jnp.zeros(
                (state_dim + 2 * action_dim + 2 * constraint_dim) * horizon
            ),
            lower_bound=jnp.zeros(
                (
                    state_dim
                    + (horizon - 1) * state_dim
                    + horizon * action_dim
                    + horizon * constraint_dim
                    + 2 * horizon * constraint_dim
                )
            ),
            upper_bound=jnp.zeros(
                (
                    state_dim
                    + (horizon - 1) * state_dim
                    + horizon * action_dim
                    + horizon * constraint_dim
                    + 2 * horizon * constraint_dim
                )
            ),
            state_dim=state_dim,
            action_dim=action_dim,
            constraint_dim=constraint_dim,
            horizon=horizon,
            policy_graphdef=policy_graphdef,
            relax_to_go_graphdef=relax_to_go_graphdef,
            sys_dynamics=sys_dynamics,
            sys_constraints=sys_constraints,
        )

    def constraint_fn(
        self, x: jax.Array, phi_d: ProjectionPhiD, phi_nd: ProjectionPhiND
    ) -> jax.Array:
        # Mock implementation for testing purposes
        return jnp.zeros(
            (
                self.state_dim
                + (self.horizon - 1) * self.state_dim
                + self.horizon * self.action_dim
                + self.horizon * self.constraint_dim
                + 2 * self.horizon * self.constraint_dim
            )
        )

    def x_to_rollout(self, x: jax.Array):
        # Mock implementation for testing purposes
        return Rollout(
            states=jnp.zeros((self.horizon, self.state_dim)),
            actions=jnp.zeros((self.horizon, self.action_dim)),
            interventions=jnp.zeros((self.horizon, self.action_dim)),
            relaxations_plus=jnp.zeros((self.horizon, self.constraint_dim)),
            relaxations_minus=jnp.zeros((self.horizon, self.constraint_dim)),
        )


class TestProjectionBase:
    """Test class for ProjectionBase."""

    state_dim: int = 3
    action_dim: int = 2
    constraint_dim: int = 5
    horizon: int = 10

    @pytest.fixture
    def policy(self) -> PolicyBase:
        return PolicyMock.factory(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )

    @pytest.fixture
    def sys_dynamics(self) -> SystemDynamicsBase:
        return SystemDynamicsMock(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )

    @pytest.fixture
    def sys_constraints(self) -> SystemConstraintsBase:
        return SystemConstraintsMock.factory(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )

    @pytest.fixture
    def relax_to_go(self):
        return DropoutMLP(
            in_features=self.state_dim,
            out_features=1,
            hidden_features=[64],
            rngs=nnx.Rngs(42),
        )

    @pytest.fixture
    def projection(
        self,
        policy,
        relax_to_go,
        sys_dynamics,
        sys_constraints,
    ) -> ProjectionMock:
        return ProjectionMock.factory(
            policy=policy,
            relax_to_go=relax_to_go,
            sys_dynamics=sys_dynamics,
            sys_constraints=sys_constraints,
            horizon=self.horizon,
        )

    @pytest.fixture
    def phi_d(
        self,
        policy,
        relax_to_go,
    ) -> ProjectionPhiD:
        return ProjectionPhiD(
            policy_params=nnx.state(policy, nnx.Param),
            relax_to_go_params=nnx.state(relax_to_go, nnx.Param),
        )

    @pytest.fixture
    def phi_nd(self, policy, relax_to_go) -> ProjectionPhiND:
        return ProjectionPhiND(
            policy_non_params=nnx.state(policy, nnx.Not(nnx.Param)),
            relax_to_go_non_params=nnx.state(relax_to_go, nnx.Not(nnx.Param)),
            initial_state=jnp.zeros(self.state_dim),
            jax_key=jax.random.PRNGKey(0),
        )

    def test_initial_state_constraints(self, projection, phi_d, phi_nd):
        # Setup
        x = projection.get_x0(phi_d, phi_nd)
        rollout = projection.x_to_rollout(x)

        # Check JIT call
        constr = jax.jit(projection.constraints_initial_state)(
            rollout, phi_nd.initial_state
        ).block_until_ready()
        constr_ref = rollout.states[0] - phi_nd.initial_state
        assert jnp.allclose(constr, constr_ref)

        # Check VJP
        test_util.check_vjp(
            f=projection.constraints_initial_state,
            f_vjp=partial(jax.vjp, projection.constraints_initial_state),
            args=(rollout, phi_nd.initial_state),
            rtol=1 / 100,
        )

        # Check JVP
        test_util.check_jvp(
            f=projection.constraints_initial_state,
            f_jvp=partial(jax.jvp, projection.constraints_initial_state),
            args=(rollout, phi_nd.initial_state),
            rtol=1 / 100,
        )

        # Check bounds call
        bounds_shape = projection.bounds_initial_state(self.state_dim)
        assert bounds_shape[0].shape == bounds_shape[1].shape
        assert constr.shape == bounds_shape[0].shape

    def test_sys_dynamics_constraints(self, projection, phi_d, phi_nd):
        # Setup
        x = projection.get_x0(phi_d, phi_nd)
        rollout = projection.x_to_rollout(x)

        # Check JIT call
        constr = jax.jit(projection.constraints_sys_dynamics)(
            rollout, projection.sys_dynamics
        ).block_until_ready()
        constr_ref = (
            rollout.states[1:]
            - projection.sys_dynamics(rollout.states[:-1], rollout.actions[:-1])
        ).ravel()
        assert jnp.allclose(constr, constr_ref)

        # Capture nondiff params
        captured = partial(
            projection.constraints_sys_dynamics,
            sys_dynamics=projection.sys_dynamics,
        )

        # Check constraints VJP
        test_util.check_vjp(
            f=captured,
            f_vjp=partial(jax.vjp, captured),
            args=(rollout,),
            rtol=1 / 100,
        )

        # Check constraints JVP
        test_util.check_jvp(
            f=captured,
            f_jvp=partial(jax.jvp, captured),
            args=(rollout,),
            rtol=1 / 100,
        )

        # Check bounds call
        bounds_shape = projection.bounds_sys_dynamics(
            self.horizon, self.state_dim
        )
        assert bounds_shape[0].shape == bounds_shape[1].shape
        assert constr.shape == bounds_shape[0].shape

    def test_policy_constraints(self, projection, phi_d, phi_nd, policy):
        # Setup
        x = projection.get_x0(phi_d, phi_nd)
        rollout = projection.x_to_rollout(x)

        # Check JIT call
        constr = jax.jit(projection.constraints_policy)(
            rollout, policy, phi_nd.jax_key
        ).block_until_ready()
        constr_ref = (
            rollout.actions
            - policy(rollout.states, jax_key=phi_nd.jax_key)
            - rollout.interventions
        ).ravel()
        assert jnp.allclose(constr, constr_ref)

        # Capture nondiff params
        captured = partial(
            projection.constraints_policy,
            policy=policy,
            jax_key=phi_nd.jax_key,
        )

        # Check VJP
        test_util.check_vjp(
            f=captured,
            f_vjp=partial(jax.vjp, captured),
            args=(rollout,),
            rtol=1 / 100,
        )

        # Check JVP
        test_util.check_jvp(
            f=captured,
            f_jvp=partial(jax.jvp, captured),
            args=(rollout,),
            rtol=1 / 100,
        )

        # Check bounds call
        bounds_shape = projection.bounds_policy(self.horizon, self.action_dim)
        assert bounds_shape[0].shape == bounds_shape[1].shape
        assert constr.shape == bounds_shape[0].shape

    def test_sys_constraints(self, projection, phi_d, phi_nd, sys_constraints):
        # Setup
        x = projection.get_x0(phi_d, phi_nd)
        rollout = projection.x_to_rollout(x)

        # Check JIT call
        constr = jax.jit(projection.constraints_sys_constraints)(
            rollout, sys_constraints
        ).block_until_ready()
        constr_ref = (
            sys_constraints(rollout.states, rollout.actions)
            + rollout.relaxations_plus
            - rollout.relaxations_minus
        ).ravel()
        assert jnp.allclose(constr, constr_ref)

        # Check VJP
        def constraint_fn(r):
            return projection.constraints_sys_constraints(r, sys_constraints)

        test_util.check_vjp(
            f=constraint_fn,
            f_vjp=partial(jax.vjp, constraint_fn),
            args=(rollout,),
            rtol=1 / 100,
        )

        # Check JVP
        test_util.check_jvp(
            f=constraint_fn,
            f_jvp=partial(jax.jvp, constraint_fn),
            args=(rollout,),
            rtol=1 / 100,
        )

        # Additional checks
        assert jnp.all(
            sys_constraints.lower_bound[None, :]
            <= constr.reshape(self.horizon, -1)
        ), "Lower bound of system constraints not satisfied."
        assert jnp.all(
            constr.reshape(self.horizon, -1)
            <= sys_constraints.upper_bound[None, :]
        ), "Upper bound of system constraints not satisfied."

        # Check bounds call
        bounds_shape = projection.bounds_sys_constraints(
            self.horizon, sys_constraints
        )
        assert bounds_shape[0].shape == bounds_shape[1].shape
        assert constr.shape == bounds_shape[0].shape

    def test_non_neg_relaxations(self, projection, phi_d, phi_nd):
        # Setup
        x = projection.get_x0(phi_d, phi_nd)
        rollout = projection.x_to_rollout(x)

        # Check JIT call
        constr = jax.jit(projection.constraints_non_neg_relax)(
            rollout
        ).block_until_ready()
        constr_ref = jnp.concatenate(
            [rollout.relaxations_plus, rollout.relaxations_minus]
        ).ravel()
        assert jnp.allclose(constr, constr_ref)

        # Check VJP
        test_util.check_vjp(
            f=projection.constraints_non_neg_relax,
            f_vjp=partial(jax.vjp, projection.constraints_non_neg_relax),
            args=(rollout,),
            rtol=1 / 100,
        )

        # Check JVP
        test_util.check_jvp(
            f=projection.constraints_non_neg_relax,
            f_jvp=partial(jax.jvp, projection.constraints_non_neg_relax),
            args=(rollout,),
            rtol=1 / 100,
        )

        # Check bounds call
        bounds_shape = projection.bounds_non_neg_relax(
            self.horizon, self.constraint_dim
        )
        assert bounds_shape[0].shape == bounds_shape[1].shape
        assert constr.shape == bounds_shape[0].shape

    def test_get_x0(self, projection, phi_d, phi_nd):
        x = jax.jit(projection.get_x0)(phi_d, phi_nd).block_until_ready()
        assert projection.cost_quad_diag.shape == x.shape
        assert projection.cost_linear.shape == x.shape

        captured = partial(projection.get_x0, phi_nd=phi_nd)

        # Check VJP
        test_util.check_vjp(
            f=captured,
            f_vjp=partial(jax.vjp, captured),
            args=(phi_d,),
            rtol=1 / 100,
        )

        # Check JVP
        test_util.check_jvp(
            f=captured,
            f_jvp=partial(jax.jvp, captured),
            args=(phi_d,),
            rtol=1 / 100,
        )

    def test_constraint_fn(self, projection, phi_d, phi_nd):
        x = projection.get_x0(phi_d, phi_nd)
        c = jax.jit(projection.constraint_fn)(
            x, phi_d, phi_nd
        ).block_until_ready()
        assert projection.lower_bound.shape == c.shape
        assert projection.upper_bound.shape == c.shape

        captured = partial(projection.constraint_fn, phi_nd=phi_nd)

        # Check VJP
        test_util.check_vjp(
            f=captured,
            f_vjp=partial(jax.vjp, captured),
            args=(x, phi_d),
            rtol=1 / 100,
        )

        # Check JVP
        test_util.check_jvp(
            f=captured,
            f_jvp=partial(jax.jvp, captured),
            args=(x, phi_d),
            rtol=1 / 100,
        )
