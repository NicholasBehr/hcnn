from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from hcnn.optimization_layer import Rollout
from hcnn.state_to_rollout.state_to_rollout_base import StateToRolloutBase
from hcnn.system_dynamics import SystemDynamicsBase
from hcnn.training import DropoutMLP


@dataclass(frozen=True)
class StateToRolloutDense(StateToRolloutBase):
    rollout_estimator: DropoutMLP
    sys_dynamics: SystemDynamicsBase

    @classmethod
    def factory(
        cls,
        state_dim: int,
        action_dim: int,
        constr_dim: int,
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
        ## UNIFIED ROLLOUT ESTIMATOR ##
        # Input: state
        rollout_in_feat = state_dim

        # Output: states, actions, and relaxation variables over horizon
        state_out_feat = state_dim * horizon
        action_out_feat = action_dim * horizon
        rollout_out_feat = state_out_feat + action_out_feat

        # Set up output bounds
        rollout_out_min = None
        rollout_out_max = None

        # Apply action bounds if provided
        if action_min is not None or action_max is not None:
            rollout_out_min = jnp.full(rollout_out_feat, -jnp.inf)
            rollout_out_max = jnp.full(rollout_out_feat, jnp.inf)

            # Action bounds apply to the action portion of the output
            action_start_idx = state_out_feat
            action_end_idx = state_out_feat + action_out_feat

            if action_min is not None:
                action_bounds_min = jnp.tile(action_min, horizon)
                rollout_out_min = rollout_out_min.at[
                    action_start_idx:action_end_idx
                ].set(action_bounds_min)

            if action_max is not None:
                action_bounds_max = jnp.tile(action_max, horizon)
                rollout_out_max = rollout_out_max.at[
                    action_start_idx:action_end_idx
                ].set(action_bounds_max)

        rollout_mlp = DropoutMLP(
            in_features=rollout_in_feat,
            out_features=rollout_out_feat,
            hidden_features=hidden_features,
            rngs=rngs,
            activation=activation,
            dropout_rate=dropout_rate,
            periodic_in_features=periodic_states,
            out_feature_min=rollout_out_min,
            out_feature_max=rollout_out_max,
        )

        return StateToRolloutDense(
            state_dim=state_dim,
            action_dim=action_dim,
            constr_dim=constr_dim,
            horizon=horizon,
            rollout_estimator=rollout_mlp,
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
        """
        # Get batch dimensions for reshaping
        batch_shape = state.shape[
            :-1
        ]  # All dimensions except the last (state_dim)

        # Get unified rollout prediction (outputs states and actions)
        rollout_flat = self.rollout_estimator(state)

        # Calculate output dimensions
        state_out_feat = self.state_dim * self.horizon
        action_out_feat = self.action_dim * self.horizon

        # Split the output into states and actions
        states_flat = rollout_flat[..., :state_out_feat]
        action_end_idx = state_out_feat + action_out_feat
        actions_flat = rollout_flat[..., state_out_feat:action_end_idx]

        # Reshape to proper dimensions
        states = states_flat.reshape(
            *batch_shape,
            self.horizon,
            self.state_dim,
        )
        actions = actions_flat.reshape(
            *batch_shape,
            self.horizon,
            self.action_dim,
        )

        return Rollout(states=states, actions=actions)
