import jax
from flax import nnx, struct

from hcnn.training.dropout_mlp import DropoutMLP

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
        action_range: tuple[float, float] | None = None,
    ) -> "PolicyMLP":
        dropout_mlp = DropoutMLP(
            in_features=state_dim,
            out_features=action_dim,
            hidden_features=hidden_features,
            rngs=rngs,
            activation=activation,
            dropout_rate=dropout_rate,
            periodic_in_features=periodic_states,
            out_feature_range=action_range,
        )
        return PolicyMLP(
            state_dim=state_dim,
            action_dim=action_dim,
            dropout_mlp=dropout_mlp,
        )

    def __call__(
        self,
        state: jax.Array,
        *,
        jax_key: jax.Array | None = None,
    ) -> jax.Array:
        rngs = None if jax_key is None else nnx.Rngs(jax_key)
        return self.dropout_mlp(x=state, rngs=rngs)
