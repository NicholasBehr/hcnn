from typing import List

import jax.numpy as jnp
from flax import nnx


class DropoutMLP(nnx.Module):
    """A simple Multi-Layer Perceptron (MLP) module with Dropout."""

    layers: List[nnx.Linear | nnx.Dropout | nnx.Module]
    rngs: nnx.Rngs

    def __init__(
        self,
        layer_sizes: list[int],
        dropout_rate: float = 0.0,
        activation: nnx.Module = nnx.celu,
        *,
        rngs: nnx.Rngs,
    ):
        """Initializes the MLP module.

        Args:
            layer_sizes: A list of integers specifying the number of units
                in each layer. The last element is the output size.
            dropout_rate: The dropout rate to apply after each hidden layer.
                Must be in range [0.0, 1.0].
                Defaults to 0.0 (no dropout).
            activation: The activation function to use after each hidden layer.
                Defaults to nnx.celu.
            rngs: NNX Rngs object for parameter initialization and dropout.
        """
        if dropout_rate < 0.0 or dropout_rate > 1.0:
            raise ValueError("dropout_rate must be in range [0.0, 1.0].")
        self.rngs = rngs

        self.layers = []
        for i, size in enumerate(layer_sizes[:-1]):
            # Create a Linear layer
            self.layers.append(nnx.Linear(size, layer_sizes[i + 1], rngs=rngs))

            # Add activation after each hidden layer
            if i < len(layer_sizes) - 2:
                self.layers.append(activation)

                # Add Dropout layer after the activation in hidden layers
                if dropout_rate > 0.0:
                    self.layers.append(nnx.Dropout(dropout_rate))

    def __call__(
        self, x: jnp.ndarray, *, rngs: nnx.Rngs | None = None
    ) -> jnp.ndarray:
        """Performs the forward pass through the MLP.

        Args:
            x: The input tensor.
            rngs: Optional NNX Rngs object.

        Returns:
            The output tensor after passing through the MLP.
        """
        # if no rngs are provided use internal ones
        if rngs is None:
            rngs = self.rngs

        # iteratively apply all layers
        for layer in self.layers:
            if isinstance(layer, nnx.Dropout):
                x = layer(x, rngs=rngs)
            else:
                x = layer(x)

        return x
