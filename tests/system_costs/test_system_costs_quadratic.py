import jax
import jax.numpy as jnp
import pytest

from hcnn.system_costs import SystemCostsQuadratic

from .test_system_costs_base import TestSystemCostBase


class TestSystemCostsQuadratic(TestSystemCostBase):
    """Test class for SystemCostQuadratic."""

    @pytest.fixture
    def sys_cost(self) -> SystemCostsQuadratic:
        key = jax.random.key(0)
        state_cost_root = jax.random.normal(
            key, (self.state_dim, self.state_dim)
        )
        action_cost_root = jax.random.normal(
            key, (self.action_dim, self.action_dim)
        )
        return SystemCostsQuadratic.factory(
            state_cost_matrix=state_cost_root.T @ state_cost_root,
            action_cost_matrix=action_cost_root.T @ action_cost_root,
        )

    def test_instance_creation_quadratic(self, sys_cost: SystemCostsQuadratic):
        assert isinstance(sys_cost, SystemCostsQuadratic)
        assert sys_cost.state_cost_matrix.shape == (
            self.state_dim,
            self.state_dim,
        )
        assert sys_cost.action_cost_matrix.shape == (
            self.action_dim,
            self.action_dim,
        )
        assert sys_cost.reference_state.shape == (self.state_dim,)
        assert sys_cost.reference_action.shape == (self.action_dim,)
        assert sys_cost.periodic_states is None  # Default should be None

    def test_call_example_single(self, sys_cost: SystemCostsQuadratic):
        instance = SystemCostsQuadratic.factory(
            state_cost_matrix=jnp.array([[1.0, 2.0], [2.0, 3.0]]),
            action_cost_matrix=jnp.array([4.0]),
        )
        state = jnp.array([1.0, 2.0])
        action = jnp.array([3.0])

        cost = instance(state, action)

        # With zero reference: (state - 0)^T Q (state - 0) +
        # (action - 0)^T R (action - 0)
        # state_cost = [1, 2] @ [[1, 2], [2, 3]] @ [1, 2]^T
        # = [5, 8] @ [1, 2]^T = 21
        # action_cost = [3] @ [4] @ [3]^T = 36
        # total = 21 + 36 = 57
        expected = jnp.array(57.0)
        assert jnp.allclose(cost, expected)

    def test_call_example_batch(self):
        instance = SystemCostsQuadratic.factory(
            state_cost_matrix=jnp.array([[1.0, 2.0], [2.0, 3.0]]),
            action_cost_matrix=jnp.array([4.0]),
        )
        state = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        action = jnp.array([[3.0], [4.0]])

        cost = instance(state, action)

        # For first sample: state_cost = 21, action_cost = 36, total = 57
        # For second sample: [3, 4] @ Q @ [3, 4]^T + [4] @ R @ [4]^T
        # state_cost = [11, 18] @ [3, 4]^T = 105, action_cost = 64
        # total = 105 + 64 = 169
        expected = jnp.array([57.0, 169.0])
        assert jnp.allclose(cost, expected)

    def test_call_with_reference_state_action(self):
        """Test cost computation with non-zero reference state and action."""
        instance = SystemCostsQuadratic.factory(
            state_cost_matrix=jnp.array([[1.0, 0.0], [0.0, 1.0]]),  # Identity
            action_cost_matrix=jnp.array([[2.0]]),
            reference_state=jnp.array([1.0, 1.0]),
            reference_action=jnp.array([0.5]),
        )
        # Test single sample
        state = jnp.array([2.0, 3.0])  # deviation = [1.0, 2.0]
        action = jnp.array([1.5])  # deviation = [1.0]

        cost = instance(state, action)

        # state_cost = [1, 2] @ I @ [1, 2]^T = 1 + 4 = 5
        # action_cost = [1] @ [2] @ [1]^T = 2
        # total = 5 + 2 = 7
        expected = jnp.array(7.0)
        assert jnp.allclose(cost, expected)

    def test_call_batch_with_reference(self):
        """Test batch computation with non-zero reference."""
        instance = SystemCostsQuadratic.factory(
            state_cost_matrix=jnp.array([[1.0, 0.0], [0.0, 1.0]]),  # Identity
            action_cost_matrix=jnp.array([[1.0]]),
            reference_state=jnp.array([1.0, 1.0]),
            reference_action=jnp.array([0.0]),
        )

        # Batch of states and actions
        states = jnp.array([[1.0, 1.0], [2.0, 3.0], [0.0, 0.0]])
        actions = jnp.array([[0.0], [1.0], [2.0]])

        costs = instance(states, actions)

        # Sample 1: deviation = [0, 0], cost = 0 + 0 = 0
        # Sample 2: deviation = [1, 2], cost = 1 + 4 + 1 = 6
        # Sample 3: deviation = [-1, -1], cost = 1 + 1 + 4 = 6
        expected = jnp.array([0.0, 6.0, 6.0])
        assert jnp.allclose(costs, expected)

    def test_reference_defaults_to_zero(self):
        """Test that default reference is zero vector."""
        instance = SystemCostsQuadratic.factory(
            state_cost_matrix=jnp.array([[1.0]]),
            action_cost_matrix=jnp.array([[1.0]]),
        )

        assert jnp.allclose(instance.reference_state, jnp.array([0.0]))
        assert jnp.allclose(instance.reference_action, jnp.array([0.0]))

    def test_multi_batch_dimensions(self):
        """Test with multiple batch dimensions."""
        instance = SystemCostsQuadratic.factory(
            state_cost_matrix=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
            action_cost_matrix=jnp.array([[1.0]]),
            reference_state=jnp.array([0.0, 0.0]),
            reference_action=jnp.array([0.0]),
        )

        # Shape: (batch_size=2, horizon=3, state_dim=2)
        states = jnp.array(
            [
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                [[2.0, 0.0], [0.0, 2.0], [1.0, 2.0]],
            ]
        )
        # Shape: (batch_size=2, horizon=3, action_dim=1)
        actions = jnp.array([[[1.0], [1.0], [1.0]], [[2.0], [2.0], [2.0]]])

        costs = instance(states, actions)

        # Should have shape (batch_size=2, horizon=3)
        assert costs.shape == (2, 3)

        # Verify some calculations
        # First trajectory, first step: [1,0] -> cost = 1,
        # action [1] -> cost = 1, total = 2
        assert jnp.allclose(costs[0, 0], 2.0)
        # Second trajectory, first step: [2,0] -> cost = 4,
        # action [2] -> cost = 4, total = 8
        assert jnp.allclose(costs[1, 0], 8.0)

    def test_periodic_states_basic(self):
        """Test basic periodic state functionality."""
        instance = SystemCostsQuadratic.factory(
            state_cost_matrix=jnp.array([[1.0, 0.0], [0.0, 1.0]]),  # Identity
            action_cost_matrix=jnp.array([[1.0]]),
            reference_state=jnp.array([0.0, 0.0]),
            reference_action=jnp.array([0.0]),
            periodic_states=frozenset([0]),  # First state is periodic
        )

        # Test wrapping: π and -π should give same cost
        state1 = jnp.array([jnp.pi, 1.0])
        state2 = jnp.array([-jnp.pi, 1.0])
        action = jnp.array([0.0])

        cost1 = instance(state1, action)
        cost2 = instance(state2, action)

        # Both should give same cost since π ≡ -π (mod 2π)
        assert jnp.allclose(cost1, cost2)

    def test_periodic_states_wrapping(self):
        """Test that periodic states wrap correctly."""
        instance = SystemCostsQuadratic.factory(
            state_cost_matrix=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
            action_cost_matrix=jnp.array([[1.0]]),
            reference_state=jnp.array([0.0, 0.0]),
            reference_action=jnp.array([0.0]),
            periodic_states=frozenset([0]),
        )

        # Test that 2π wraps to 0
        state1 = jnp.array([2 * jnp.pi, 0.0])
        state2 = jnp.array([0.0, 0.0])
        action = jnp.array([0.0])

        cost1 = instance(state1, action)
        cost2 = instance(state2, action)

        assert jnp.allclose(cost1, cost2, atol=1e-6)

    def test_periodic_states_multiple_indices(self):
        """Test with multiple periodic state indices."""
        instance = SystemCostsQuadratic.factory(
            state_cost_matrix=jnp.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            action_cost_matrix=jnp.array([[1.0]]),
            reference_state=jnp.array([0.0, 0.0, 0.0]),
            reference_action=jnp.array([0.0]),
            periodic_states=frozenset([0, 2]),  # First and third periodic
        )

        # Test with periodic wrapping on multiple dimensions
        state = jnp.array([2 * jnp.pi, 1.0, -jnp.pi])  # Wrap 0th and 2nd
        action = jnp.array([0.0])

        cost = instance(state, action)

        # Should be equivalent to [0, 1, π] due to wrapping
        # Cost = 0 + 1 + π² = 1 + π²
        expected_cost = 1.0 + jnp.pi**2
        assert jnp.allclose(cost, expected_cost, atol=1e-6)

    def test_periodic_states_with_reference(self):
        """Test periodic states with non-zero reference."""
        instance = SystemCostsQuadratic.factory(
            state_cost_matrix=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
            action_cost_matrix=jnp.array([[1.0]]),
            reference_state=jnp.array([jnp.pi / 2, 1.0]),  # π/2 ref for angle
            reference_action=jnp.array([0.0]),
            periodic_states=frozenset([0]),
        )

        # State at -3π/2 should wrap to π/2, giving zero deviation
        state = jnp.array([-3 * jnp.pi / 2, 1.0])
        action = jnp.array([0.0])

        cost = instance(state, action)

        # Should be 0 cost since -3π/2 ≡ π/2 (mod 2π)
        assert jnp.allclose(cost, 0.0, atol=1e-6)

    def test_periodic_states_batch(self):
        """Test periodic states with batch dimensions."""
        instance = SystemCostsQuadratic.factory(
            state_cost_matrix=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
            action_cost_matrix=jnp.array([[1.0]]),
            reference_state=jnp.array([0.0, 0.0]),
            reference_action=jnp.array([0.0]),
            periodic_states=frozenset([0]),
        )

        # Batch of states with periodic wrapping
        states = jnp.array(
            [
                [0.0, 1.0],  # No wrapping needed
                [2 * jnp.pi, 2.0],  # 2π wraps to 0
                [-jnp.pi, 0.5],  # -π stays -π
            ]
        )
        actions = jnp.array([[0.0], [0.0], [0.0]])

        costs = instance(states, actions)

        # Expected costs: [1, 4, π² + 0.25]
        expected = jnp.array([1.0, 4.0, jnp.pi**2 + 0.25])
        assert jnp.allclose(costs, expected, atol=1e-6)

    def test_periodic_states_validation(self):
        """Test validation of periodic state indices."""
        # Should raise assertion error for out-of-range index
        with pytest.raises(AssertionError):
            SystemCostsQuadratic.factory(
                state_cost_matrix=jnp.array([[1.0]]),
                action_cost_matrix=jnp.array([[1.0]]),
                periodic_states=frozenset([1]),  # Index 1 out of range
            )

    def test_no_periodic_states_default(self):
        """Test that periodic_states defaults to None."""
        instance = SystemCostsQuadratic.factory(
            state_cost_matrix=jnp.array([[1.0]]),
            action_cost_matrix=jnp.array([[1.0]]),
        )
        assert instance.periodic_states is None
