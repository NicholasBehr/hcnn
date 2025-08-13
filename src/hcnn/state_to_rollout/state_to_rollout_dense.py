import jax
import jax.numpy as jnp
from flax import nnx, struct

from hcnn.projection import Rollout
from hcnn.system_dynamics import SystemDynamicsBase
from hcnn.training import DropoutMLP

from .state_to_rollout_base import StateToRolloutBase


@struct.dataclass
class StateToRolloutDense(StateToRolloutBase):
    state_estimator: DropoutMLP
    action_estimator: DropoutMLP
    relax_estimator: DropoutMLP
    sys_dynamics: SystemDynamicsBase

    @classmethod
    def factory(
        cls,
        state_dim: int,
        action_dim: int,
        hidden_features: list[int],
        horizon: int,
        sys_dynamics: SystemDynamicsBase,
        *,
        rngs: nnx.Rngs,
        activation: str = "leaky_relu",
        dropout_rate: float = 0.0,
        periodic_states: frozenset[int] | None = None,
        action_min: jnp.ndarray | None = None,
        action_max: jnp.ndarray | None = None,
    ) -> "StateToRolloutDense":
        ## STATE ESTIMATOR ##
        state_in_feat = state_dim
        state_out_feat = state_dim * horizon
        state_out_min = None
        state_out_max = None

        state_mlp = DropoutMLP(
            in_features=state_in_feat,
            out_features=state_out_feat,
            hidden_features=hidden_features,
            rngs=rngs,
            activation=activation,
            dropout_rate=dropout_rate,
            periodic_in_features=periodic_states,
            out_feature_min=state_out_min,
            out_feature_max=state_out_max,
        )

        ## ACTION ESTIMATOR ##
        action_in_feat = state_dim
        action_out_feat = action_dim * horizon
        action_out_min = None
        action_out_max = None
        if action_min is not None:
            action_out_min = jnp.tile(action_min, horizon)
        if action_max is not None:
            action_out_max = jnp.tile(action_max, horizon)

        action_mlp = DropoutMLP(
            in_features=action_in_feat,
            out_features=action_out_feat,
            hidden_features=hidden_features,
            rngs=rngs,
            activation=activation,
            dropout_rate=dropout_rate,
            periodic_in_features=periodic_states,
            out_feature_min=action_out_min,
            out_feature_max=action_out_max,
        )

        ## RELAXATION ESTIMATOR ##
        relax_in_feat = state_dim
        relax_out_feat = state_dim * horizon
        relax_out_min = None
        relax_out_max = None

        relax_mlp = DropoutMLP(
            in_features=relax_in_feat,
            out_features=relax_out_feat,
            hidden_features=hidden_features,
            rngs=rngs,
            activation=activation,
            dropout_rate=dropout_rate,
            periodic_in_features=periodic_states,
            out_feature_min=relax_out_min,
            out_feature_max=relax_out_max,
        )

        return StateToRolloutDense(
            state_dim=state_dim,
            action_dim=action_dim,
            horizon=horizon,
            state_estimator=state_mlp,
            action_estimator=action_mlp,
            relax_estimator=relax_mlp,
            sys_dynamics=sys_dynamics,
        )

    def __call__(self, state: jax.Array) -> Rollout:
        """Estimates the rollout given the current state.

        Args:
            state (jax.Array): Current state, shape (..., state_dim)

        Returns:
            Estimated rollout with states and actions
                states, shape (..., horizon, state_dim)
                actions, shape (..., horizon, action_dim)
                states_relax, shape (..., horizon, state_dim)
        """
        # Get batch dimensions for reshaping
        batch_shape = state.shape[
            :-1
        ]  # All dimensions except the last (state_dim)

        # Estimate states over horizon
        states_flat = self.state_estimator(state)
        states = states_flat.reshape(*batch_shape, self.horizon, self.state_dim)

        # Estimate actions over horizon
        actions_flat = self.action_estimator(state)
        actions = actions_flat.reshape(
            *batch_shape, self.horizon, self.action_dim
        )

        # Estimate relaxed states over horizon
        states_relax_flat = self.relax_estimator(state)
        states_relax = states_relax_flat.reshape(
            *batch_shape, self.horizon, self.state_dim
        )

        return Rollout(
            states=states, actions=actions, states_relax=states_relax
        )
