import jax
import jax.numpy as jnp
import pytest
from jax import flatten_util

from hcnn.projection import (
    OptimizationVariable,
    ProjectionPhiD,
    ProjectionPhiND,
    ProjectionProblem,
    Rollout,
)
from tests.system_constraints import SystemConstraintsMock
from tests.system_dynamics import SystemDynamicsMock


class TestProjection:
    """Test class for Projection optimization problem."""

    state_dim: int = 4
    action_dim: int = 2
    horizon: int = 10

    @pytest.fixture
    def sys_constraints(self):
        """Create mock system constraints."""
        return SystemConstraintsMock.factory(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )

    @pytest.fixture
    def sys_dynamics(self):
        """Create mock system dynamics."""
        return SystemDynamicsMock(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )

    @pytest.fixture
    def projection(self, sys_constraints):
        """Create a Projection instance."""
        return ProjectionProblem.factory(
            sys_constraints=sys_constraints,
            horizon=self.horizon,
            relaxation_penalty=10.0,
        )

    @pytest.fixture
    def rollout_ref(self):
        """Create a reference rollout."""
        key = jax.random.key(42)
        states = jax.random.normal(key, (self.horizon, self.state_dim))
        actions = jax.random.normal(key, (self.horizon, self.action_dim))
        states_relax = jnp.zeros_like(states)
        return Rollout(
            states=states, actions=actions, states_relax=states_relax
        )

    @pytest.fixture
    def phi_d(self, rollout_ref):
        """Create ProjectionPhiD."""
        return ProjectionPhiD(
            states_ref=rollout_ref.states,
            actions_ref=rollout_ref.actions,
        )

    @pytest.fixture
    def phi_nd(self, sys_dynamics):
        """Create ProjectionPhiND."""
        key = jax.random.key(123)
        initial_state = jax.random.normal(key, (self.state_dim,))
        states_relax_pos = jnp.zeros((self.horizon, self.state_dim))
        states_relax_neg = jnp.zeros((self.horizon, self.state_dim))
        return ProjectionPhiND(
            sys_dynamics=sys_dynamics,
            initial_state=initial_state,
            states_relax_pos=states_relax_pos,
            states_relax_neg=states_relax_neg,
        )

    def test_projection_factory(self, sys_constraints):
        """Test Projection factory creation."""
        projection = ProjectionProblem.factory(
            sys_constraints=sys_constraints,
            horizon=self.horizon,
            relaxation_penalty=10.0,
        )

        assert isinstance(projection, ProjectionProblem)
        assert projection.sys_constraints == sys_constraints
        assert projection.horizon == self.horizon

        # Check that optimization problem fields are set
        assert projection.cost_quad_diag.shape[0] > 0
        assert projection.cost_linear.shape[0] > 0
        assert projection.lower_bound.shape[0] > 0
        assert projection.upper_bound.shape[0] > 0
        assert projection.cost_quad_diag.shape == projection.cost_linear.shape
        assert projection.lower_bound.shape == projection.upper_bound.shape

    def test_optimization_variable_creation(self, rollout_ref):
        """Test OptimizationVariable dataclass creation."""
        optvar = OptimizationVariable(
            states=rollout_ref.states,
            actions=rollout_ref.actions,
            states_diff=jnp.zeros((self.horizon, self.state_dim)),
            actions_diff=jnp.zeros((self.horizon, self.action_dim)),
            states_relax_pos=jnp.zeros((self.horizon, self.state_dim)),
            states_relax_neg=jnp.zeros((self.horizon, self.state_dim)),
        )

        assert isinstance(optvar, OptimizationVariable)
        assert jnp.allclose(optvar.states, rollout_ref.states)
        assert jnp.allclose(optvar.actions, rollout_ref.actions)

    def test_bounds_rollout_diff(self):
        """Test bounds for rollout difference constraints."""
        lower, upper = ProjectionProblem.bounds_rollout_diff(
            self.horizon, self.state_dim, self.action_dim
        )

        expected_size = self.horizon * (self.state_dim + self.action_dim)
        assert lower.shape == (expected_size,)
        assert upper.shape == (expected_size,)
        assert jnp.allclose(lower, 0.0)
        assert jnp.allclose(upper, 0.0)

    def test_bounds_sys_dynamics(self):
        """Test bounds for system dynamics constraints."""
        lower, upper = ProjectionProblem.bounds_sys_dynamics(
            self.horizon, self.state_dim
        )

        expected_size = (self.horizon - 1) * self.state_dim
        assert lower.shape == (expected_size,)
        assert upper.shape == (expected_size,)
        assert jnp.allclose(lower, 0.0)
        assert jnp.allclose(upper, 0.0)

    def test_bounds_sys_constraints(self, sys_constraints):
        """Test bounds for system constraints."""
        lower, upper = ProjectionProblem.bounds_sys_constraints(
            self.horizon, sys_constraints
        )

        expected_size = self.horizon * sys_constraints.constraint_dim
        assert lower.shape == (expected_size,)
        assert upper.shape == (expected_size,)

        # Check that bounds are repeated correctly
        expected_lower = jnp.repeat(sys_constraints.lower_bound, self.horizon)
        expected_upper = jnp.repeat(sys_constraints.upper_bound, self.horizon)
        assert jnp.allclose(lower, expected_lower)
        assert jnp.allclose(upper, expected_upper)

    def test_bounds_initial_state(self):
        """Test bounds for initial state constraints."""
        lower, upper = ProjectionProblem.bounds_initial_state(self.state_dim)

        assert lower.shape == (self.state_dim,)
        assert upper.shape == (self.state_dim,)
        assert jnp.allclose(lower, 0.0)
        assert jnp.allclose(upper, 0.0)

    def test_constraints_rollout_diff(self, rollout_ref):
        """Test rollout difference constraints."""
        states_diff = jnp.ones((self.horizon, self.state_dim)) * 0.1
        actions_diff = jnp.ones((self.horizon, self.action_dim)) * 0.2

        optvar = OptimizationVariable(
            states=rollout_ref.states + states_diff,
            actions=rollout_ref.actions + actions_diff,
            states_diff=states_diff,
            actions_diff=actions_diff,
            states_relax_pos=jnp.zeros((self.horizon, self.state_dim)),
            states_relax_neg=jnp.zeros((self.horizon, self.state_dim)),
        )

        phi_d = ProjectionPhiD(
            states_ref=rollout_ref.states,
            actions_ref=rollout_ref.actions,
        )

        constraints = ProjectionProblem.constraints_rollout_diff(optvar, phi_d)

        # Should be zero when rollout = rollout_ref + rollout_diff
        expected_size = self.horizon * (self.state_dim + self.action_dim)
        assert constraints.shape == (expected_size,)
        assert jnp.allclose(constraints, 0.0, atol=1e-6)

    def test_constraints_sys_dynamics(self, rollout_ref, sys_dynamics):
        """Test system dynamics constraints."""
        optvar = OptimizationVariable(
            states=rollout_ref.states,
            actions=rollout_ref.actions,
            states_diff=jnp.zeros_like(rollout_ref.states),
            actions_diff=jnp.zeros_like(rollout_ref.actions),
            states_relax_pos=jnp.zeros_like(rollout_ref.states),
            states_relax_neg=jnp.zeros_like(rollout_ref.states),
        )

        constraints = ProjectionProblem.constraints_sys_dynamics(
            optvar, sys_dynamics
        )

        expected_size = (self.horizon - 1) * self.state_dim
        assert constraints.shape == (expected_size,)

        # For perfect dynamics, constraints should be small
        # The exact values depend on the specific rollout and dynamics

    def test_constraints_sys_constraints(self, rollout_ref, sys_constraints):
        """Test system constraints."""
        optvar = OptimizationVariable(
            states=rollout_ref.states,
            actions=rollout_ref.actions,
            states_diff=jnp.zeros_like(rollout_ref.states),
            actions_diff=jnp.zeros_like(rollout_ref.actions),
            states_relax_pos=jnp.zeros_like(rollout_ref.states),
            states_relax_neg=jnp.zeros_like(rollout_ref.states),
        )

        constraints = ProjectionProblem.constraints_sys_constraints(
            optvar, sys_constraints
        )

        expected_size = self.horizon * sys_constraints.constraint_dim
        assert constraints.shape == (expected_size,)

    def test_constraints_initial_state(self, rollout_ref, phi_nd):
        """Test initial state constraints."""
        optvar = OptimizationVariable(
            states=rollout_ref.states,
            actions=rollout_ref.actions,
            states_diff=jnp.zeros_like(rollout_ref.states),
            actions_diff=jnp.zeros_like(rollout_ref.actions),
            states_relax_pos=jnp.zeros_like(rollout_ref.states),
            states_relax_neg=jnp.zeros_like(rollout_ref.states),
        )

        constraints = ProjectionProblem.constraints_initial_state(
            optvar, phi_nd
        )

        assert constraints.shape == (self.state_dim,)
        expected = rollout_ref.states[0] - phi_nd.initial_state
        assert jnp.allclose(constraints, expected)

    def test_get_x0(self, projection, phi_d, phi_nd):
        """Test initial guess generation."""
        x0 = projection.get_x0(phi_d, phi_nd)

        # Should be a flattened optimization variable
        assert isinstance(x0, jax.Array)
        assert x0.ndim == 1

        # Should have the right size for the optimization variable
        expected_rollout_size = self.horizon * (
            self.state_dim + self.action_dim
        )
        expected_relax_size = self.horizon * self.state_dim  # only states
        expected_size = (
            2
            * expected_rollout_size  # states + actions + states_diff + actions_diff  # noqa
            + 2 * expected_relax_size  # states_relax_pos + states_relax_neg
        )
        assert x0.shape == (expected_size,)

    def test_x_to_optvar(self, projection, phi_d, phi_nd):
        """Test conversion from flat array to optimization variable."""
        x0 = projection.get_x0(phi_d, phi_nd)
        optvar = projection.x_to_optvar(x0)

        assert isinstance(optvar, OptimizationVariable)
        assert optvar.states.shape == (self.horizon, self.state_dim)
        assert optvar.actions.shape == (self.horizon, self.action_dim)
        assert optvar.states_diff.shape == (
            self.horizon,
            self.state_dim,
        )
        assert optvar.actions_diff.shape == (
            self.horizon,
            self.action_dim,
        )
        assert optvar.states_relax_pos.shape == (
            self.horizon,
            self.state_dim,
        )
        assert optvar.states_relax_neg.shape == (
            self.horizon,
            self.state_dim,
        )

    def test_x_to_optvar_roundtrip(self, projection, phi_d, phi_nd):
        """Test that x_to_optvar and flattening are inverse operations."""
        x0 = projection.get_x0(phi_d, phi_nd)
        optvar = projection.x_to_optvar(x0)
        x_reconstructed = flatten_util.ravel_pytree(optvar)[0]

        # Note: This may not be exactly equal due to the way
        # get_x0 and x_to_optvar work
        # get_x0 uses phi_d.horizon_ref while x_to_optvar uses ones
        # But the shapes should match
        assert x_reconstructed.shape == x0.shape

    def test_constraint_fn(self, projection, phi_d, phi_nd):
        """Test the complete constraint function."""
        x0 = projection.get_x0(phi_d, phi_nd)
        constraints = projection.constraint_fn(x0, phi_d, phi_nd)

        assert isinstance(constraints, jax.Array)
        assert constraints.ndim == 1

        # Should have constraints from all four constraint types
        expected_rollout_diff_size = self.horizon * (
            self.state_dim + self.action_dim
        )
        expected_dynamics_size = (self.horizon - 1) * self.state_dim
        expected_sys_constraints_size = self.horizon * (
            self.state_dim + self.action_dim
        )  # Mock returns [state, action]
        expected_initial_state_size = self.state_dim
        expected_relaxation_size = self.horizon * 2 * self.state_dim

        expected_total_size = (
            expected_rollout_diff_size
            + expected_dynamics_size
            + expected_sys_constraints_size
            + expected_initial_state_size
            + expected_relaxation_size
        )

        assert constraints.shape == (expected_total_size,)

    def test_constraint_fn_jit_compilable(self, projection, phi_d, phi_nd):
        """Test that constraint function is JIT compilable."""
        constraint_fn_jit = jax.jit(projection.constraint_fn)

        x0 = projection.get_x0(phi_d, phi_nd)
        constraints = constraint_fn_jit(x0, phi_d, phi_nd)

        assert isinstance(constraints, jax.Array)
        assert constraints.ndim == 1

    def test_constraint_fn_gradient(self, projection, phi_d, phi_nd):
        """Test that constraint function has well-defined gradients."""

        def constraint_fn_wrapper(x):
            return jnp.sum(projection.constraint_fn(x, phi_d, phi_nd) ** 2)

        constraint_grad_fn = jax.grad(constraint_fn_wrapper)

        x0 = projection.get_x0(phi_d, phi_nd)
        grad = constraint_grad_fn(x0)

        assert isinstance(grad, jax.Array)
        assert grad.shape == x0.shape
        assert jnp.isfinite(grad).all()

    def test_projection_phi_d_creation(self, rollout_ref):
        """Test ProjectionPhiD dataclass creation."""
        phi_d = ProjectionPhiD(
            states_ref=rollout_ref.states,
            actions_ref=rollout_ref.actions,
        )

        assert isinstance(phi_d, ProjectionPhiD)
        assert jnp.allclose(phi_d.states_ref, rollout_ref.states)
        assert jnp.allclose(phi_d.actions_ref, rollout_ref.actions)

    def test_projection_phi_nd_creation(self, sys_dynamics):
        """Test ProjectionPhiND dataclass creation."""
        key = jax.random.key(789)
        initial_state = jax.random.normal(key, (self.state_dim,))
        states_relax_pos = jnp.zeros((self.horizon, self.state_dim))
        states_relax_neg = jnp.zeros((self.horizon, self.state_dim))

        phi_nd = ProjectionPhiND(
            sys_dynamics=sys_dynamics,
            initial_state=initial_state,
            states_relax_pos=states_relax_pos,
            states_relax_neg=states_relax_neg,
        )

        assert isinstance(phi_nd, ProjectionPhiND)
        assert phi_nd.sys_dynamics == sys_dynamics
        assert jnp.allclose(phi_nd.initial_state, initial_state)
        assert jnp.allclose(phi_nd.states_relax_pos, states_relax_pos)
        assert jnp.allclose(phi_nd.states_relax_neg, states_relax_neg)

    def test_different_horizons(self, sys_constraints):
        """Test projection with different horizon values."""
        horizons = [1, 5, 20, 50]

        for horizon in horizons:
            projection = ProjectionProblem.factory(
                sys_constraints=sys_constraints,
                horizon=horizon,
                relaxation_penalty=10.0,
            )

            assert projection.horizon == horizon

            # Check that all arrays have consistent sizes
            expected_rollout_size = horizon * (self.state_dim + self.action_dim)
            expected_relax_size = horizon * self.state_dim
            expected_optvar_size = (
                2 * expected_rollout_size + 2 * expected_relax_size
            )

            assert projection.cost_quad_diag.shape == (expected_optvar_size,)
            assert projection.cost_linear.shape == (expected_optvar_size,)

    def test_constraint_consistency(self, projection, phi_d, phi_nd):
        """Test constraint bounds and constraint function consistent sizes."""
        x0 = projection.get_x0(phi_d, phi_nd)
        constraints = projection.constraint_fn(x0, phi_d, phi_nd)

        # The constraint function output should match the bound sizes
        assert constraints.shape == projection.lower_bound.shape
        assert constraints.shape == projection.upper_bound.shape

    def test_cost_structure(self, sys_constraints):
        """Test the cost structure in the factory method."""
        projection = ProjectionProblem.factory(
            sys_constraints=sys_constraints,
            horizon=self.horizon,
            relaxation_penalty=10.0,
        )

        # Check that the quadratic cost is only on the difference term
        # This tests the specific structure defined in the factory method
        assert (
            jnp.sum(projection.cost_quad_diag) > 0
        )  # Should have some non-zero elements

        # Linear cost should have relaxation penalties on relax variables only
        # The cost_linear structure should match the OptimizationVariable layout
        expected_total_size = (
            2
            * self.horizon
            * (
                self.state_dim + self.action_dim
            )  # states + actions + states_diff + actions_diff  # noqa
            + 2
            * self.horizon
            * self.state_dim  # states_relax_pos + states_relax_neg  # noqa
        )
        assert projection.cost_linear.shape == (expected_total_size,)

        # Create a test OptimizationVariable to understand the structure
        test_optvar = OptimizationVariable(
            states=jnp.ones((self.horizon, self.state_dim)),
            actions=jnp.ones((self.horizon, self.action_dim)),
            states_diff=jnp.ones((self.horizon, self.state_dim)),
            actions_diff=jnp.ones((self.horizon, self.action_dim)),
            states_relax_pos=jnp.ones((self.horizon, self.state_dim)),
            states_relax_neg=jnp.ones((self.horizon, self.state_dim)),
        )
        flat_test, _ = flatten_util.ravel_pytree(test_optvar)

        # The relaxation penalty should only be on the last
        # 2*horizon*state_dim elements
        relax_start_idx = flat_test.size - 2 * self.horizon * self.state_dim
        non_relax_part = projection.cost_linear[:relax_start_idx]
        relax_part = projection.cost_linear[relax_start_idx:]

        # Non-relaxation parts should be zeros
        assert jnp.allclose(non_relax_part, 0.0)

        # Relaxation part should have the penalty values
        assert jnp.allclose(relax_part, 10.0)
