import jax
import jax.numpy as jnp
import pytest
from flax import nnx, struct
from jax import flatten_util

from hcnn.projection.projection_base import (
    ProjectionBase,
    ProjectionPhi,
    Rollout,
)
from hcnn.projection.projection_no_subs_no_term import (
    ProjectionNoSubstitutionNoTerminal,
)
from tests.policy.test_policy_base import PolicyBase, PolicyMock
from tests.system_constraints.test_system_constraints_base import (
    SystemConstraintsBase,
    SystemConstraintsMock,
)
from tests.system_dynamics.test_system_dynamics_base import (
    SystemDynamicsBase,
    SystemDynamicsMock,
)


@struct.dataclass
class ProjectionMock(ProjectionBase):
    @classmethod
    def factory(
        cls,
        policy,
        sys_dynamics,
        sys_constraints,
        horizon,
    ) -> "ProjectionMock":
        state_dim = sys_dynamics.state_dim
        action_dim = sys_dynamics.action_dim
        constraint_dim = sys_constraints.constraint_dim
        policy_graphdef = nnx.split(policy)[0]
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
            sys_dynamics=sys_dynamics,
            sys_constraints=sys_constraints,
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

    def get_x0(self, phi: ProjectionPhi) -> jax.Array:
        # Mock implementation for testing purposes
        rollout = Rollout(
            states=jnp.zeros((self.horizon, self.state_dim)),
            actions=jnp.zeros((self.horizon, self.action_dim)),
            interventions=jnp.zeros((self.horizon, self.action_dim)),
            relaxations_plus=jnp.zeros((self.horizon, self.constraint_dim)),
            relaxations_minus=jnp.zeros((self.horizon, self.constraint_dim)),
        )
        return flatten_util.ravel_pytree(rollout)[0]

    def constraint_fn(self, x: jax.Array, phi: ProjectionPhi) -> jax.Array:
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


class TestProjectionBase:
    """Test class for ProjectionBase."""

    @pytest.fixture
    def policy(self, state_dim, action_dim) -> PolicyBase:
        return PolicyMock.factory(
            state_dim=state_dim,
            action_dim=action_dim,
        )

    @pytest.fixture
    def sys_dynamics(self, state_dim, action_dim) -> SystemDynamicsBase:
        return SystemDynamicsMock(
            state_dim=state_dim,
            action_dim=action_dim,
        )

    @pytest.fixture
    def sys_constraints(self, state_dim, action_dim) -> SystemConstraintsBase:
        return SystemConstraintsMock.factory(
            state_dim=state_dim,
            action_dim=action_dim,
        )

    @pytest.fixture
    def projection(
        self,
        policy,
        sys_dynamics,
        sys_constraints,
        horizon,
    ) -> ProjectionMock:
        return ProjectionMock.factory(
            policy=policy,
            sys_dynamics=sys_dynamics,
            sys_constraints=sys_constraints,
            horizon=horizon,
        )

    @pytest.fixture
    def phi(
        self,
        policy,
        state_dim,
    ) -> ProjectionPhi:
        keys = jax.random.split(jax.random.key(0), 2)
        return ProjectionPhi.factory(
            initial_state=jax.random.normal(keys[0], (state_dim,)),
            policy=policy,
            jax_key=keys[1],
        )

    @pytest.fixture(autouse=True)
    def setup(
        self,
        policy: PolicyBase,
        sys_dynamics: SystemDynamicsBase,
        sys_constraints: SystemConstraintsBase,
        projection: ProjectionNoSubstitutionNoTerminal,
        phi: ProjectionPhi,
    ):
        self.state_dim = projection.state_dim
        self.action_dim = projection.action_dim
        self.constraint_dim = projection.constraint_dim
        self.horizon = projection.horizon
        self.policy = policy
        self.sys_dynamics = sys_dynamics
        self.sys_constraints = sys_constraints
        self.projection = projection
        self.phi = phi

    def test_problem(self):
        x = self.projection.get_x0(self.phi)
        assert self.projection.cost_quad_diag.shape == x.shape
        assert self.projection.cost_linear.shape == x.shape

        c = self.projection.constraint_fn(x, self.phi)
        assert self.projection.lower_bound.shape == c.shape
        assert self.projection.upper_bound.shape == c.shape

    def test_initial_state_constraints(self):
        x = self.projection.get_x0(self.phi)
        rollout = self.projection.x_to_rollout(x)

        # Check initial state constraints
        initial_state_constraints = self.projection.constraints_initial_state(
            rollout, self.phi.initial_state
        )
        ref_initial_state_constraints = (
            rollout.states[0] - self.phi.initial_state
        )

        assert jnp.allclose(
            initial_state_constraints, ref_initial_state_constraints
        ), "Initial state constraints do not match expected values."

        bounds_shape = self.projection.bounds_initial_state(self.state_dim)
        assert bounds_shape[0].shape == bounds_shape[1].shape
        assert initial_state_constraints.shape == bounds_shape[0].shape

    def test_sys_dynamics_constraints(self):
        x = self.projection.get_x0(self.phi)
        rollout = self.projection.x_to_rollout(x)

        # Check system dynamics constraints
        sys_dynamics_constraints = self.projection.constraints_sys_dynamics(
            rollout, self.sys_dynamics
        )
        ref_sys_dynamics_constraints = (
            rollout.states[1:]
            - self.sys_dynamics(rollout.states[:-1], rollout.actions[:-1])
        ).ravel()

        assert jnp.allclose(
            sys_dynamics_constraints, ref_sys_dynamics_constraints
        ), "System dynamics constraints do not match expected values."

        bounds_shape = self.projection.bounds_sys_dynamics(
            self.horizon, self.state_dim
        )
        assert bounds_shape[0].shape == bounds_shape[1].shape
        assert sys_dynamics_constraints.shape == bounds_shape[0].shape

    def test_policy_constraints(self):
        x = self.projection.get_x0(self.phi)
        rollout = self.projection.x_to_rollout(x)

        # Check policy constraints
        policy_constraints = self.projection.constraints_policy(
            rollout, self.policy, self.phi.jax_key
        )
        ref_policy_constraints = (
            rollout.actions
            - self.policy(rollout.states, jax_key=self.phi.jax_key)
            - rollout.interventions
        ).ravel()

        assert jnp.allclose(policy_constraints, ref_policy_constraints), (
            "Policy constraints do not match expected values."
        )

        bounds_shape = self.projection.bounds_policy(
            self.horizon, self.action_dim
        )
        assert bounds_shape[0].shape == bounds_shape[1].shape
        assert policy_constraints.shape == bounds_shape[0].shape

    def test_sys_constraints(self):
        x = self.projection.get_x0(self.phi)
        rollout = self.projection.x_to_rollout(x)

        # Check system constraints
        sys_constraints = self.projection.constraints_sys_constraints(
            rollout, self.sys_constraints
        )
        ref_sys_constraints = (
            self.sys_constraints(rollout.states, rollout.actions)
            + rollout.relaxations_plus
            - rollout.relaxations_minus
        ).ravel()
        assert jnp.allclose(sys_constraints, ref_sys_constraints), (
            "System constraints do not match expected values."
        )

        assert jnp.all(
            self.sys_constraints.lower_bound[None, :]
            <= sys_constraints.reshape(self.horizon, -1)
        ), "Lower bound of system constraints not satisfied."
        assert jnp.all(
            sys_constraints.reshape(self.horizon, -1)
            <= self.sys_constraints.upper_bound[None, :]
        ), "Upper bound of system constraints not satisfied."

        bounds_shape = self.projection.bounds_sys_constraints(
            self.horizon, self.sys_constraints
        )
        assert bounds_shape[0].shape == bounds_shape[1].shape
        assert sys_constraints.shape == bounds_shape[0].shape

    def test_non_neg_relaxations(self):
        x = self.projection.get_x0(self.phi)
        rollout = self.projection.x_to_rollout(x)

        non_neg_relax_constraints = self.projection.constraints_non_neg_relax(
            rollout
        )
        ref_non_neg_relax_constraints = jnp.concatenate(
            [rollout.relaxations_plus, rollout.relaxations_minus]
        ).ravel()
        assert jnp.allclose(
            non_neg_relax_constraints, ref_non_neg_relax_constraints
        ), "Non-negative relaxations constraints do not match expected values."

        bounds_shape = self.projection.bounds_non_neg_relax(
            self.horizon, self.constraint_dim
        )
        assert bounds_shape[0].shape == bounds_shape[1].shape
        assert non_neg_relax_constraints.shape == bounds_shape[0].shape
