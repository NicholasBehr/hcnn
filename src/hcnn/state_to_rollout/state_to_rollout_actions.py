import jax
import jax.numpy as jnp
from flax import nnx, struct

from hcnn.projection import Rollout
from hcnn.system_dynamics import SystemDynamicsBase
from hcnn.training import DropoutMLP

from .state_to_rollout_base import StateToRolloutBase


@struct.dataclass
class StateToRolloutActions(StateToRolloutBase):
    action_estimator: DropoutMLP
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
    ) -> "StateToRolloutActions":
        action_in_feat = state_dim
        action_out_feat = action_dim * horizon
        action_out_min = None
        action_out_max = None
        if action_min is not None:
            action_out_min = jnp.tile(action_min, horizon)
        if action_max is not None:
            action_out_max = jnp.tile(action_max, horizon)

        dropout_mlp = DropoutMLP(
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

        return StateToRolloutActions(
            state_dim=state_dim,
            action_dim=action_dim,
            constr_dim=constr_dim,
            horizon=horizon,
            action_estimator=dropout_mlp,
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
        actions_flat = self.action_estimator(state)
        # Reshape actions from (..., horizon * action_dim) to
        # (..., horizon, action_dim)
        actions_shape = actions_flat.shape
        batch_shape = actions_shape[:-1]  # All dimensions except the last
        actions = actions_flat.reshape(
            *batch_shape, self.horizon, self.action_dim
        )

        rollout = Rollout.from_actions(
            actions=actions,
            sys_dynamics=self.sys_dynamics,
            initial_state=state,
        )
        return rollout
