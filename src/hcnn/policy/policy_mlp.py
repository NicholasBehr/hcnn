import jax
from flax import nnx, struct

from hcnn.policy import PolicyBase
from hcnn.training import DropoutMLP


@struct.dataclass
class PolicyMLP(PolicyBase):
    dropout_mlp: DropoutMLP

    @classmethod
    def factory(
        cls,
        state_dim: int,
        action_dim: int,
        hidden_layer_sizes: list[int],
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ) -> "PolicyMLP":
        layer_sizes = [state_dim] + hidden_layer_sizes + [action_dim]
        dropout_mlp = DropoutMLP(layer_sizes=layer_sizes, dropout_rate=dropout_rate, rngs=rngs)
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
