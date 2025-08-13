import jax
import jax.numpy as jnp
import pytest
from flax import struct

from hcnn.projection import Rollout
from hcnn.system_dynamics import SystemDynamicsBase
from tests.system_dynamics import SystemDynamicsMock


@struct.dataclass
class LinearDynamics(SystemDynamicsBase):
    """Linear dynamics system for predictable testing."""

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        # Simple linear dynamics: next_state = A * state + B * action
        state_matrix = jnp.eye(self.state_dim) * 0.9  # slight decay
        action_matrix = jnp.ones((self.state_dim, self.action_dim)) * 0.1
        return state_matrix @ state + action_matrix @ action


class TestRollout:
    """Test class for Rollout dataclass and its methods."""

    state_dim: int = 4
    action_dim: int = 2
    horizon: int = 10
    batch_size: int = 5

    @pytest.fixture
    def dynamics(self) -> SystemDynamicsBase:
        """Create a simple mock dynamics for testing."""
        return SystemDynamicsMock(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )

    @pytest.fixture
    def linear_dynamics(self) -> SystemDynamicsBase:
        """Create a linear dynamics system for more predictable testing."""
        return LinearDynamics(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )

    def test_rollout_creation(self):
        """Test basic Rollout dataclass creation."""
        key = jax.random.key(42)
        states = jax.random.normal(key, (self.horizon, self.state_dim))
        actions = jax.random.normal(key, (self.horizon, self.action_dim))
        states_relax = jnp.zeros_like(states)

        rollout = Rollout(
            states=states, actions=actions, states_relax=states_relax
        )

        assert rollout.states.shape == (self.horizon, self.state_dim)
        assert rollout.actions.shape == (self.horizon, self.action_dim)
        assert rollout.states_relax.shape == (self.horizon, self.state_dim)
        assert jnp.allclose(rollout.states, states)
        assert jnp.allclose(rollout.actions, actions)
        assert jnp.allclose(rollout.states_relax, states_relax)

    def test_rollout_properties(self):
        """Test the state_dim and action_dim properties."""
        key = jax.random.key(42)
        states = jax.random.normal(key, (self.horizon, self.state_dim))
        actions = jax.random.normal(key, (self.horizon, self.action_dim))
        states_relax = jnp.zeros_like(states)

        rollout = Rollout(
            states=states, actions=actions, states_relax=states_relax
        )

        assert rollout.state_dim == self.state_dim
        assert rollout.action_dim == self.action_dim

    def test_rollout_properties_batched(self):
        """Test properties work with batched rollouts."""
        key = jax.random.key(42)
        batch_size = 3
        states = jax.random.normal(
            key, (batch_size, self.horizon, self.state_dim)
        )
        actions = jax.random.normal(
            key, (batch_size, self.horizon, self.action_dim)
        )
        states_relax = jnp.zeros_like(states)

        rollout = Rollout(
            states=states, actions=actions, states_relax=states_relax
        )

        # Properties should still return the last dimension
        assert rollout.state_dim == self.state_dim
        assert rollout.action_dim == self.action_dim

    def test_from_actions_single_trajectory(self, linear_dynamics):
        """Test from_actions with a single trajectory (no batch dimension)."""
        key = jax.random.key(42)
        initial_state = jax.random.normal(key, (self.state_dim,))
        actions = jax.random.normal(key, (self.horizon, self.action_dim))

        rollout = Rollout.from_actions(
            actions=actions,
            sys_dynamics=linear_dynamics,
            initial_state=initial_state,
        )

        # Check shapes
        assert rollout.states.shape == (self.horizon, self.state_dim)
        assert rollout.actions.shape == (self.horizon, self.action_dim)
        assert jnp.allclose(rollout.actions, actions)

        # Verify dynamics are applied correctly
        # states[0] should be the initial state before first action
        # states[1] should be the result of applying actions[0] to states[0]
        expected_state_1 = linear_dynamics(rollout.states[0], actions[0])
        assert jnp.allclose(rollout.states[1], expected_state_1, atol=1e-6)

    def test_from_actions_batched_trajectory(self, linear_dynamics):
        """Test from_actions with batched trajectories."""
        key = jax.random.key(42)
        initial_state = jax.random.normal(
            key, (self.batch_size, self.state_dim)
        )
        actions = jax.random.normal(
            key, (self.batch_size, self.horizon, self.action_dim)
        )

        rollout = Rollout.from_actions(
            actions=actions,
            sys_dynamics=linear_dynamics,
            initial_state=initial_state,
        )

        # Check shapes
        expected_states_shape = (self.batch_size, self.horizon, self.state_dim)
        expected_actions_shape = (
            self.batch_size,
            self.horizon,
            self.action_dim,
        )
        assert rollout.states.shape == expected_states_shape
        assert rollout.actions.shape == expected_actions_shape
        assert jnp.allclose(rollout.actions, actions)

        # Verify dynamics for first batch element
        expected_state_1 = linear_dynamics(rollout.states[0, 0], actions[0, 0])
        assert jnp.allclose(rollout.states[0, 1], expected_state_1, atol=1e-6)

    def test_from_actions_multi_batch_dimensions(self, linear_dynamics):
        """Test from_actions with multiple batch dimensions."""
        key = jax.random.key(42)
        batch_shape = (3, 4)  # Multiple batch dimensions

        initial_state = jax.random.normal(key, (*batch_shape, self.state_dim))
        actions = jax.random.normal(
            key, (*batch_shape, self.horizon, self.action_dim)
        )

        rollout = Rollout.from_actions(
            actions=actions,
            sys_dynamics=linear_dynamics,
            initial_state=initial_state,
        )

        # Check shapes
        expected_states_shape = (*batch_shape, self.horizon, self.state_dim)
        expected_actions_shape = (*batch_shape, self.horizon, self.action_dim)

        assert rollout.states.shape == expected_states_shape
        assert rollout.actions.shape == expected_actions_shape
        assert jnp.allclose(rollout.actions, actions)

        # Verify dynamics for one element
        expected_state_1 = linear_dynamics(
            rollout.states[0, 0, 0], actions[0, 0, 0]
        )
        assert jnp.allclose(
            rollout.states[0, 0, 1], expected_state_1, atol=1e-6
        )

    def test_from_actions_state_sequence_property(self, linear_dynamics):
        """Test that states represent the sequence correctly.

        states[t] should be the state before applying actions[t],
        which transitions to states[t+1].
        """
        key = jax.random.key(42)
        initial_state = jax.random.normal(key, (self.state_dim,))
        actions = jax.random.normal(key, (self.horizon, self.action_dim))

        rollout = Rollout.from_actions(
            actions=actions,
            sys_dynamics=linear_dynamics,
            initial_state=initial_state,
        )

        # Manually compute expected states sequence
        expected_states = []
        current_state = initial_state
        for t in range(self.horizon):
            expected_states.append(current_state)
            current_state = linear_dynamics(current_state, actions[t])

        expected_states = jnp.stack(expected_states)
        assert jnp.allclose(rollout.states, expected_states, atol=1e-6)

    def test_from_actions_zero_horizon(self, linear_dynamics):
        """Test from_actions with zero horizon."""
        key = jax.random.key(42)
        initial_state = jax.random.normal(key, (self.state_dim,))
        actions = jnp.empty((0, self.action_dim))  # Zero horizon

        rollout = Rollout.from_actions(
            actions=actions,
            sys_dynamics=linear_dynamics,
            initial_state=initial_state,
        )

        assert rollout.states.shape == (0, self.state_dim)
        assert rollout.actions.shape == (0, self.action_dim)

    def test_from_actions_single_timestep(self, linear_dynamics):
        """Test from_actions with single timestep."""
        key = jax.random.key(42)
        initial_state = jax.random.normal(key, (self.state_dim,))
        actions = jax.random.normal(key, (1, self.action_dim))

        rollout = Rollout.from_actions(
            actions=actions,
            sys_dynamics=linear_dynamics,
            initial_state=initial_state,
        )

        assert rollout.states.shape == (1, self.state_dim)
        assert rollout.actions.shape == (1, self.action_dim)
        assert jnp.allclose(rollout.states[0], initial_state)

    def test_from_actions_jit_compilation(self, linear_dynamics):
        """Test that from_actions can be JIT compiled."""
        key = jax.random.key(42)
        initial_state = jax.random.normal(key, (self.state_dim,))
        actions = jax.random.normal(key, (self.horizon, self.action_dim))

        @jax.jit
        def jitted_from_actions(actions, initial_state):
            return Rollout.from_actions(
                actions=actions,
                sys_dynamics=linear_dynamics,
                initial_state=initial_state,
            )

        # Should not raise any errors
        rollout = jitted_from_actions(actions, initial_state)
        assert rollout.states.shape == (self.horizon, self.state_dim)
        assert rollout.actions.shape == (self.horizon, self.action_dim)

    def test_from_actions_vmap_compatibility(self, linear_dynamics):
        """Test that from_actions works with vmap over different axes."""
        key = jax.random.key(42)

        # Test vmapping over different initial states
        initial_states = jax.random.normal(
            key, (self.batch_size, self.state_dim)
        )
        actions = jax.random.normal(key, (self.horizon, self.action_dim))

        @jax.vmap
        def vmap_from_actions(initial_state):
            return Rollout.from_actions(
                actions=actions,
                sys_dynamics=linear_dynamics,
                initial_state=initial_state,
            )

        rollouts = vmap_from_actions(initial_states)
        expected_states_shape = (self.batch_size, self.horizon, self.state_dim)
        expected_actions_shape = (
            self.batch_size,
            self.horizon,
            self.action_dim,
        )
        assert rollouts.states.shape == expected_states_shape
        assert rollouts.actions.shape == expected_actions_shape

    def test_from_actions_gradient_compatibility(self, linear_dynamics):
        """Test that gradients can be computed through from_actions."""
        key = jax.random.key(42)
        initial_state = jax.random.normal(key, (self.state_dim,))
        actions = jax.random.normal(key, (self.horizon, self.action_dim))

        def loss_fn(initial_state):
            rollout = Rollout.from_actions(
                actions=actions,
                sys_dynamics=linear_dynamics,
                initial_state=initial_state,
            )
            return jnp.sum(rollout.states**2)

        # Should not raise any errors
        loss_value = loss_fn(initial_state)
        grad_fn = jax.grad(loss_fn)
        gradients = grad_fn(initial_state)

        assert gradients.shape == initial_state.shape
        assert jnp.isfinite(loss_value).all()
        assert jnp.isfinite(gradients).all()

    def test_from_actions_different_dynamics(self, dynamics):
        """Test from_actions with the simple mock dynamics."""
        key = jax.random.key(42)
        initial_state = jax.random.normal(key, (self.state_dim,))
        actions = jax.random.normal(key, (self.horizon, self.action_dim))

        rollout = Rollout.from_actions(
            actions=actions,
            sys_dynamics=dynamics,
            initial_state=initial_state,
        )

        # With SystemDynamicsMock, each state should be the same as the
        # initial state since the mock dynamics just returns the input state
        expected_states = jnp.tile(initial_state, (self.horizon, 1))
        assert jnp.allclose(rollout.states, expected_states)

    def test_from_actions_precision(self, linear_dynamics):
        """Test numerical precision of the state evolution."""
        # Use simple initial conditions for precise testing
        initial_state = jnp.ones(self.state_dim)
        actions = jnp.zeros((self.horizon, self.action_dim))

        rollout = Rollout.from_actions(
            actions=actions,
            sys_dynamics=linear_dynamics,
            initial_state=initial_state,
        )

        # With zero actions and our linear dynamics (A = 0.9*I, B = 0.1*ones),
        # states should decay by factor of 0.9 each step
        for t in range(self.horizon):
            expected_state = initial_state * (0.9**t)
            assert jnp.allclose(rollout.states[t], expected_state, atol=1e-10)
