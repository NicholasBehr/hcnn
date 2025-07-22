from typing import Any, List

import jax.numpy as jnp
from flax import nnx

activation_map = {
    "celu": nnx.celu,
    "relu": nnx.relu,
    "leaky_relu": nnx.leaky_relu,
    "tanh": nnx.tanh,
    "hard_sigmoid": nnx.hard_sigmoid,
}


class DropoutMLP(nnx.Module):
    """A Multi-Layer Perceptron (MLP) module with Dropout."""

    layers: List[nnx.Linear | nnx.Dropout | Any]
    rngs: nnx.Rngs
    periodic_in_features: set[int]
    out_feature_range: tuple[float, float] | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: list[int],
        *,
        rngs: nnx.Rngs,
        activation: str = "leaky_relu",
        dropout_rate: float = 0.0,
        periodic_in_features: frozenset[int] | None = None,
        out_feature_range: tuple[float, float] | None = None,
    ):
        """Initializes the MLP module.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            hidden_features: A list of integers specifying the number of units
                in each hidden layer.
            rngs: NNX Rngs object for parameter initialization and dropout.
            activation: The activation function to use after each hidden layer.
                Defaults to nnx.celu.
            dropout_rate: The dropout rate to apply after each hidden layer.
                Must be in range [0.0, 1.0].
                Defaults to 0.0 (no dropout).
            periodic_in_features: A set of input feature indices that are
                2π periodic. These features will be transformed using sin and
                cos before being fed into the network. If None or empty, no
                periodic transformations are applied. Defaults to None.
            out_feature_range: Optional tuple (min, max) to constrain output
                features. If provided, output will be squashed using sigmoid and
                scaled to the specified range. Defaults to None.
        """
        if dropout_rate < 0.0 or dropout_rate > 1.0:
            raise ValueError("dropout_rate must be in range [0.0, 1.0].")
        self.rngs = rngs
        self.periodic_in_features = (
            periodic_in_features if periodic_in_features is not None else set()
        )
        self.out_feature_range = out_feature_range

        # Calculate the actual input size after periodic transformation
        # Each periodic feature becomes 2 features (sin and cos)
        # Non-periodic features remain 1 feature each
        num_periodic = len(self.periodic_in_features)
        transformed_input_size = in_features - num_periodic + 2 * num_periodic

        # Create layer sizes: [transformed_input] + hidden + [out_features]
        layer_sizes = (
            [transformed_input_size] + hidden_features + [out_features]
        )

        self.layers = []
        for i, size in enumerate(layer_sizes[:-1]):
            # Create a Linear layer
            self.layers.append(nnx.Linear(size, layer_sizes[i + 1], rngs=rngs))

            # Add activation after each hidden layer (not output layer)
            if i < len(layer_sizes) - 2:
                self.layers.append(activation_map[activation])

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

        # Apply periodic transformations if any periodic indices are specified
        if self.periodic_in_features:
            x = self._apply_periodic_transform(x)

        # iteratively apply all layers
        for layer in self.layers:
            if isinstance(layer, nnx.Dropout):
                x = layer(x, rngs=rngs)
            else:
                x = layer(x)

        # Apply output range scaling if specified
        if self.out_feature_range is not None:
            x = self._apply_output_scaling(x)

        return x

    def _apply_periodic_transform(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply sin/cos transformation to periodic features.

        Args:
            x: Input tensor with shape (..., input_features)

        Returns:
            Transformed tensor where periodic features are replaced with
            their sin and cos values.
        """
        if not self.periodic_in_features:
            return x

        # Create list to hold transformed features
        transformed_features = []

        # Process each feature index
        for i in range(x.shape[-1]):
            if i in self.periodic_in_features:
                # Apply sin and cos transformation to periodic features
                feature = x[..., i : i + 1]  # Keep dimension
                transformed_features.append(jnp.sin(feature))
                transformed_features.append(jnp.cos(feature))
            else:
                # Keep non-periodic features as-is
                transformed_features.append(x[..., i : i + 1])

        # Concatenate all features along the last dimension
        return jnp.concatenate(transformed_features, axis=-1)

    def _apply_output_scaling(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply output range scaling using sigmoid squashing.

        Args:
            x: Output tensor from the network.

        Returns:
            Scaled tensor with values constrained to the specified range.
        """
        if self.out_feature_range is None:
            return x

        min_val, max_val = self.out_feature_range
        # Apply sigmoid to squash to (0, 1), then scale to (min_val, max_val)
        x_sigmoid = nnx.sigmoid(x)
        # Scale from (0, 1) to (min_val, max_val)
        scaled_x = min_val + (max_val - min_val) * x_sigmoid
        return scaled_x
