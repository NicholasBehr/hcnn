import jax
import jax.numpy as jnp
from flax import nnx, struct

from hcnn.projection import Rollout
from hcnn.system_dynamics import SystemDynamicsBase
from hcnn.training import DropoutMLP

from .state_to_rollout_base import StateToRolloutBase


@struct.dataclass
class StateToRolloutScan(StateToRolloutBase):
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
    ) -> "StateToRolloutScan":
        ## ACTION ESTIMATOR ##
        action_mlp = DropoutMLP(
            in_features=state_dim,
            out_features=action_dim,
            hidden_features=hidden_features,
            rngs=rngs,
            activation=activation,
            dropout_rate=dropout_rate,
            periodic_in_features=periodic_states,
            out_feature_min=action_min,
            out_feature_max=action_max,
        )

        ## RELAXATION ESTIMATOR ##
        relax_mlp = DropoutMLP(
            in_features=state_dim,
            out_features=state_dim,
            hidden_features=hidden_features,
            rngs=rngs,
            activation=activation,
            dropout_rate=dropout_rate,
            periodic_in_features=periodic_states,
        )

        return StateToRolloutScan(
            state_dim=state_dim,
            action_dim=action_dim,
            horizon=horizon,
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
        # Current states shape is (..., state_dim)
        original_shape = state.shape
        batch_dims = original_shape[:-1]

        # Reshape to (batch_size, state_dim)
        flat_states = state.reshape(-1, self.state_dim)

        def scan_fn(carry, _):
            action_est_split, relax_est_split, system_state = carry
            action, action_est_split = nnx.call(action_est_split)(system_state)
            relax, relax_est_split = nnx.call(relax_est_split)(system_state)
            next_state = self.sys_dynamics(system_state, action)

            carry = (action_est_split, relax_est_split, next_state)
            out = (system_state, action, relax)
            return carry, out

        (action_est_split, relax_est_split, _), out = jax.lax.scan(
            scan_fn,
            init=(
                nnx.split(self.action_estimator),
                nnx.split(self.relax_estimator),
                flat_states,
            ),
            length=self.horizon,
        )

        nnx.update(self.action_estimator, action_est_split[1])
        nnx.update(self.relax_estimator, relax_est_split[1])

        # Reshape back
        states = out[0].reshape(*batch_dims, self.horizon, self.state_dim)
        actions = out[1].reshape(*batch_dims, self.horizon, self.action_dim)
        states_relax = out[2].reshape(*batch_dims, self.horizon, self.state_dim)

        return Rollout(
            states=states, actions=actions, states_relax=states_relax
        )
