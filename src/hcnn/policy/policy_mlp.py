import jax
import jax.numpy as jnp
from flax import nnx, struct

from hcnn.training import DropoutMLP

from .policy_base import PolicyBase


@struct.dataclass
class PolicyMLP(PolicyBase):
    dropout_mlp: DropoutMLP

    @classmethod
    def factory(
        cls,
        state_dim: int,
        action_dim: int,
        hidden_features: list[int],
        *,
        rngs: nnx.Rngs,
        activation: str = "leaky_relu",
        dropout_rate: float = 0.0,
        periodic_states: frozenset[int] | None = None,
        action_min: jnp.ndarray | None = None,
        action_max: jnp.ndarray | None = None,
    ) -> "PolicyMLP":
        """Factory method to create a PolicyMLP instance.

        Args:
            state_dim: Number of state features.
            action_dim: Number of action features.
            hidden_features: List of hidden layer sizes.
            rngs: NNX Rngs object for parameter initialization and dropout.
            activation: Activation function name. Defaults to "leaky_relu".
            dropout_rate: Dropout rate between 0.0 and 1.0. Defaults to 0.0.
            periodic_states: Set of state indices that are 2π periodic.
                These will be transformed using sin/cos. Defaults to None.
            action_min: Optional array of minimum values for each action.
                Must have shape (action_dim,) if provided. Defaults to None.
            action_max: Optional array of maximum values for each action.
                Must have shape (action_dim,) if provided. Defaults to None.

        Returns:
            A PolicyMLP instance.
        """
        dropout_mlp = DropoutMLP(
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
        return PolicyMLP(
            state_dim=state_dim,
            action_dim=action_dim,
            dropout_mlp=dropout_mlp,
        )

    def __call__(
        self,
        state: jax.Array,
    ) -> jax.Array:
        return self.dropout_mlp(x=state)
