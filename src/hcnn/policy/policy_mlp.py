from typing import List

import jax
from flax import nnx, struct

from .policy_base import PolicyBase


@struct.dataclass
class PolicyMLP(PolicyBase):
    layers: List[nnx.Module]

    @classmethod
    def factory(
        cls, layer_sizes: list[int], dropout_rate: float = 0.0, *, rngs: nnx.Rngs
    ) -> "PolicyMLP":
        """Initializes the MLP policy module.

        Args:
            layer_sizes: A list of integers specifying the number of units
                         in each layer. The first element hast to equal the state dimension,
                         and the the last element has to equal the action dimension.
            dropout_rate: The dropout rate to apply after each hidden layer.
                          Defaults to 0.0 (no dropout).
            rngs: NNX Rngs object for parameter initialization and dropout.
                  Requires 'params' and 'dropout' collections if dropout_rate > 0.
        """

        if dropout_rate < 0.0 or dropout_rate > 1.0:
            raise ValueError("dropout_rate must be in range [0.0, 1.0].")

        layers = []
        for i, size in enumerate(layer_sizes[:-1]):
            # Create a Linear layer
            linear_layer = nnx.Linear(size, layer_sizes[i + 1], rngs=rngs)
            layers.append(linear_layer)

            # Add activation after each hidden layer
            if i < len(layer_sizes) - 2:
                layers.append(nnx.celu)

                # Add Dropout layer after the activation in hidden layers
                if dropout_rate > 0.0:
                    dropout_layer = nnx.Dropout(dropout_rate)
                    layers.append(dropout_layer)

        return PolicyMLP(
            state_dim=layer_sizes[0],
            action_dim=layer_sizes[-1],
            layers=layers,
        )

    def __call__(
        self,
        state: jax.Array,
        *,
        jax_key: jax.Array | None = None,
    ) -> jax.Array:
        # the state is the input to the policy
        x = state

        rngs = nnx.Rngs(jax_key) if jax_key is not None else None
        for layer in self.layers:
            if isinstance(layer, nnx.Dropout):
                x = layer(x, deterministic=rngs is None, rngs=rngs)
            else:
                x = layer(x)

        # the action is the output of the policy
        action = x

        return action
