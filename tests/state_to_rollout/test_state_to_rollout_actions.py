import jax
import jax.numpy as jnp
import pytest
from flax import nnx, struct

from hcnn.projection import Rollout
from hcnn.state_to_rollout import StateToRolloutActions
from hcnn.system_dynamics import SystemDynamicsBase


@struct.dataclass
class SystemDynamicsMock(SystemDynamicsBase):
    """Mock system dynamics for testing."""

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        # Simple mock: next_state = state + action (if dimensions match)
        # If dimensions don't match, just return the state
        if state.shape[-1] == action.shape[-1]:
            return state + action
        return state


class TestStateToRolloutActions:
    """Test class for StateToRolloutActions."""

    @pytest.fixture
    def key(self):
        return jax.random.key(42)

    @pytest.fixture
    def rngs(self, key):
        return nnx.Rngs(key)

    @pytest.fixture
    def state_dim(self):
        return 4

    @pytest.fixture
    def action_dim(self):
        return 2

    @pytest.fixture
    def horizon(self):
        return 3

    @pytest.fixture
    def hidden_features(self):
        return [8, 6]

    @pytest.fixture
    def sys_dynamics(self, state_dim, action_dim):
        return SystemDynamicsMock(
            state_dim=state_dim,
            action_dim=action_dim,
        )

    @pytest.fixture
    def basic_model(
        self,
        state_dim,
        action_dim,
        hidden_features,
        horizon,
        sys_dynamics,
        rngs,
    ):
        """Basic StateToRolloutActions model without special features."""
        return StateToRolloutActions.factory(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_features=hidden_features,
            horizon=horizon,
            sys_dynamics=sys_dynamics,
            rngs=rngs,
        )

    @pytest.fixture
    def test_state(self, state_dim):
        """Sample state for testing."""
        return jnp.ones(state_dim)

    @pytest.fixture
    def batch_test_state(self, state_dim):
        """Sample batch of states for testing."""
        return jnp.ones((5, state_dim))

    def test_factory_creation(
        self, basic_model, state_dim, action_dim, horizon
    ):
        """Test basic model creation using factory method."""
        assert isinstance(basic_model, StateToRolloutActions)
        assert basic_model.state_dim == state_dim
        assert basic_model.action_dim == action_dim
        assert basic_model.horizon == horizon
        assert hasattr(basic_model, "dropout_mlp")
        assert hasattr(basic_model, "sys_dynamics")

    def test_call_single_state(
        self, basic_model, test_state, state_dim, action_dim, horizon
    ):
        """Test calling the model with a single state."""
        rollout = basic_model(test_state)

        assert isinstance(rollout, Rollout)
        assert rollout.states.shape == (horizon, state_dim)
        assert rollout.actions.shape == (horizon, action_dim)

    def test_call_batch_states(
        self, basic_model, batch_test_state, state_dim, action_dim, horizon
    ):
        """Test calling the model with a batch of states."""
        batch_size = batch_test_state.shape[0]
        rollout = basic_model(batch_test_state)

        assert isinstance(rollout, Rollout)
        assert rollout.states.shape == (batch_size, horizon, state_dim)
        assert rollout.actions.shape == (batch_size, horizon, action_dim)

    def test_factory_with_periodic_states(
        self,
        state_dim,
        action_dim,
        hidden_features,
        horizon,
        sys_dynamics,
        rngs,
    ):
        """Test factory creation with periodic states."""
        periodic_states = frozenset([0, 2])
        model = StateToRolloutActions.factory(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_features=hidden_features,
            horizon=horizon,
            sys_dynamics=sys_dynamics,
            rngs=rngs,
            periodic_states=periodic_states,
        )

        assert isinstance(model, StateToRolloutActions)
        # The periodic_in_features should be set in the MLP
        # horizon = 3, state_dim = 4
        assert model.dropout_mlp.periodic_in_features == frozenset(
            [0, 2, 4, 6, 8, 10]
        )

    def test_factory_with_action_bounds(
        self,
        state_dim,
        action_dim,
        hidden_features,
        horizon,
        sys_dynamics,
        rngs,
    ):
        """Test factory creation with action bounds."""
        action_min = jnp.array([-1.0, -2.0])
        action_max = jnp.array([1.0, 2.0])

        model = StateToRolloutActions.factory(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_features=hidden_features,
            horizon=horizon,
            sys_dynamics=sys_dynamics,
            rngs=rngs,
            action_min=action_min,
            action_max=action_max,
        )

        assert isinstance(model, StateToRolloutActions)
        assert jnp.all(
            model.dropout_mlp.out_feature_min == jnp.tile(action_min, horizon)
        )
        assert jnp.all(
            model.dropout_mlp.out_feature_max == jnp.tile(action_max, horizon)
        )

    def test_output_shapes_consistency(self, basic_model, test_state):
        """Test that output shapes are consistent with model parameters."""
        rollout = basic_model(test_state)

        # Check that rollout dimensions match model dimensions
        assert rollout.state_dim == basic_model.state_dim
        assert rollout.action_dim == basic_model.action_dim

        # Check that horizon is respected
        assert rollout.states.shape[0] == basic_model.horizon
        assert rollout.actions.shape[0] == basic_model.horizon
