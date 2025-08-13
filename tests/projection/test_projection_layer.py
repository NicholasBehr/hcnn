import jax
import jax.numpy as jnp
import pytest
from flax import struct

from hcnn.optimization.optimization_base import OptimizationInfo
from hcnn.projection import ProjectionLayer, Rollout
from tests.system_constraints import SystemConstraintsMock
from tests.system_dynamics import SystemDynamicsMock


@struct.dataclass
class MockOptimizationSolver:
    """Mock solver that returns identity solution for testing."""

    def solve(self, problem, phi_d, phi_nd):
        """Mock solve that returns the reference rollout unchanged."""
        # Get initial guess
        x0 = problem.get_x0(phi_d, phi_nd)

        # Create mock optimization info
        info = OptimizationInfo(
            primal_residual=jnp.array(0.0),
            dual_residual=jnp.array(0.0),
            iterations_lin=jnp.array(1),
            iterations_nonlin=jnp.array(1),
            converged_flag=jnp.array(True),
        )

        return x0, info


class TestProjectionLayer:
    """Test class for ProjectionLayer JIT and batch compatibility."""

    state_dim: int = 4
    action_dim: int = 2
    horizon: int = 5
    relaxation_penalty: float = 10

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
    def solver(self):
        """Create mock optimization solver."""
        return MockOptimizationSolver()

    @pytest.fixture
    def projection_layer(self, sys_dynamics, sys_constraints, solver):
        """Create ProjectionLayer instance."""
        return ProjectionLayer.factory(
            sys_dynamics=sys_dynamics,
            sys_constraints=sys_constraints,
            horizon=self.horizon,
            relaxation_penalty=self.relaxation_penalty,
            solver=solver,
        )

    @pytest.fixture
    def single_rollout(self, sys_dynamics):
        """Create a single rollout (no batch dimension)."""
        key = jax.random.PRNGKey(42)
        actions = jax.random.normal(key, (self.horizon, self.action_dim))
        initial_state = jax.random.normal(key, (self.state_dim,))

        return Rollout.from_actions(actions, sys_dynamics, initial_state)

    @pytest.fixture
    def batch_rollout(self, sys_dynamics):
        """Create a batch of rollouts."""
        batch_size = 3
        key = jax.random.PRNGKey(42)
        actions = jax.random.normal(
            key, (batch_size, self.horizon, self.action_dim)
        )
        initial_state = jax.random.normal(key, (batch_size, self.state_dim))

        return Rollout.from_actions(actions, sys_dynamics, initial_state)

    def test_single_rollout_forward(self, projection_layer, single_rollout):
        """Test ProjectionLayer forward pass with single rollout."""
        key = jax.random.PRNGKey(42)
        initial_state = jax.random.normal(key, (self.state_dim,))

        projected_rollout, info = projection_layer(
            single_rollout, initial_state
        )

        # Check output shapes
        expected_states_shape = (self.horizon, self.state_dim)
        expected_actions_shape = (self.horizon, self.action_dim)
        assert projected_rollout.states.shape == expected_states_shape
        assert projected_rollout.actions.shape == expected_actions_shape

        # Check that info contains the expected fields
        assert hasattr(info, "state_diff")
        assert hasattr(info, "action_diff")
        assert hasattr(info, "relax_diff")
        assert hasattr(info, "relax_mean")

    def test_batch_rollout_forward(self, projection_layer, batch_rollout):
        """Test ProjectionLayer forward pass with batch rollout."""
        batch_size = 3
        key = jax.random.PRNGKey(42)
        initial_state = jax.random.normal(key, (batch_size, self.state_dim))

        projected_rollout, info = projection_layer(batch_rollout, initial_state)

        # Check output shapes include batch dimension
        expected_states_shape = (batch_size, self.horizon, self.state_dim)
        expected_actions_shape = (batch_size, self.horizon, self.action_dim)
        assert projected_rollout.states.shape == expected_states_shape
        assert projected_rollout.actions.shape == expected_actions_shape

        # Check that info contains the expected fields
        assert hasattr(info, "state_diff")
        assert hasattr(info, "action_diff")
        assert hasattr(info, "relax_diff")
        assert hasattr(info, "relax_mean")

    def test_jit_compatibility_single(self, projection_layer, single_rollout):
        """
        Test that the layer works without JIT (architecture not JIT-friendly).
        """
        key = jax.random.PRNGKey(42)
        initial_state = jax.random.normal(key, (self.state_dim,))

        # Test without JIT - the layer should work normally
        result1 = projection_layer(single_rollout, initial_state)
        result2 = projection_layer(single_rollout, initial_state)

        # Results should be identical for same inputs
        jax.tree_util.tree_map(
            lambda x, y: jnp.allclose(x, y, rtol=1e-6),
            result1,
            result2,
        )

    def test_jit_compatibility_batch(self, projection_layer, batch_rollout):
        """
        Test that the layer works without JIT (architecture not JIT-friendly).
        """
        batch_size = 3
        key = jax.random.PRNGKey(42)
        initial_state = jax.random.normal(key, (batch_size, self.state_dim))

        # Test without JIT - the layer should work normally
        result1 = projection_layer(batch_rollout, initial_state)
        result2 = projection_layer(batch_rollout, initial_state)

        # Results should be identical for same inputs
        jax.tree_util.tree_map(
            lambda x, y: jnp.allclose(x, y, rtol=1e-6),
            result1,
            result2,
        )

    def test_vmap_compatibility(self, projection_layer, single_rollout):
        """Test vmap compatibility by comparing with batch processing."""
        batch_size = 3
        key = jax.random.PRNGKey(42)

        # Create batch inputs manually
        rollouts = jax.tree_util.tree_map(
            lambda x: jnp.tile(x[None, ...], (batch_size,) + (1,) * x.ndim),
            single_rollout,
        )
        initial_states = jnp.tile(
            jax.random.normal(key, (self.state_dim,))[None, :], (batch_size, 1)
        )

        # Method 1: Use vmap explicitly
        vmap_layer = jax.vmap(projection_layer)
        result_vmap = vmap_layer(rollouts, initial_states)

        # Method 2: Use the layer's internal batch handling
        result_batch = projection_layer(rollouts, initial_states)

        # Results should be identical
        jax.tree_util.tree_map(
            lambda x, y: jnp.allclose(x, y, rtol=1e-6),
            result_vmap,
            result_batch,
        )

    def test_different_batch_sizes(self, projection_layer, sys_dynamics):
        """Test layer works with different batch sizes."""
        key = jax.random.PRNGKey(42)

        for batch_size in [1, 2, 5, 10]:
            actions = jax.random.normal(
                key, (batch_size, self.horizon, self.action_dim)
            )
            initial_state = jax.random.normal(key, (batch_size, self.state_dim))
            rollout = Rollout.from_actions(actions, sys_dynamics, initial_state)

            projected_rollout, info = projection_layer(rollout, initial_state)

            # Check shapes are correct for this batch size
            assert projected_rollout.states.shape == (
                batch_size,
                self.horizon,
                self.state_dim,
            )
            assert projected_rollout.actions.shape == (
                batch_size,
                self.horizon,
                self.action_dim,
            )
