import jax.numpy as jnp
from flax import nnx

ACTIVATIONS = {
    "celu": nnx.celu,
    "relu": nnx.relu,
    "leaky_relu": nnx.leaky_relu,
    "tanh": nnx.tanh,
    "hard_sigmoid": nnx.hard_sigmoid,
}


class DropoutMLP(nnx.Module):
    """A Multi-Layer Perceptron (MLP) module with Dropout."""

    layers: nnx.List
    periodic_in_features: frozenset[int]  # indices of 2π periodic states
    out_feature_min: jnp.ndarray | None  # min values for each output feature
    out_feature_max: jnp.ndarray | None  # max values for each output feature

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
        out_feature_min: jnp.ndarray | None = None,
        out_feature_max: jnp.ndarray | None = None,
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
            out_feature_min: Optional array of minimum values for each output
                feature. If provided, must have shape (out_features,). Used
                together with out_feature_max for output scaling. If None, no
                minimum constraint is applied. Defaults to None.
            out_feature_max: Optional array of maximum values for each output
                feature. If provided, must have shape (out_features,). Used
                together with out_feature_min for output scaling. If None, no
                maximum constraint is applied. Defaults to None.
        """
        if dropout_rate < 0.0 or dropout_rate > 1.0:
            raise ValueError("dropout_rate must be in range [0.0, 1.0].")

        # Validate output feature constraints
        if out_feature_min is not None and out_feature_min.shape != (
            out_features,
        ):
            raise ValueError(
                f"out_feature_min must have shape ({out_features},), "
                f"got {out_feature_min.shape}"
            )
        if out_feature_max is not None and out_feature_max.shape != (
            out_features,
        ):
            raise ValueError(
                f"out_feature_max must have shape ({out_features},), "
                f"got {out_feature_max.shape}"
            )
        if (
            out_feature_min is not None
            and out_feature_max is not None
            and jnp.any(out_feature_min >= out_feature_max)
        ):
            raise ValueError(
                "out_feature_min must be less than out_feature_max "
                "for all features"
            )

        self.periodic_in_features = (
            periodic_in_features
            if periodic_in_features is not None
            else frozenset(set())
        )
        self.out_feature_min = out_feature_min
        self.out_feature_max = out_feature_max

        # Calculate the actual input size after periodic transformation
        # Each periodic feature becomes 2 features (sin and cos)
        # Non-periodic features remain 1 feature each
        num_periodic = len(self.periodic_in_features)
        transformed_input_size = in_features - num_periodic + 2 * num_periodic

        # Create layer sizes: [transformed_input] + hidden + [out_features]
        layer_sizes = (
            [transformed_input_size] + hidden_features + [out_features]
        )

        self.layers = nnx.List()
        for i, size in enumerate(layer_sizes[:-1]):
            # Create a Linear layer
            self.layers.append(nnx.Linear(size, layer_sizes[i + 1], rngs=rngs))

            # Add activation after each hidden layer (not output layer)
            if i < len(layer_sizes) - 2:
                self.layers.append(ACTIVATIONS[activation])

                # Add Dropout layer after the activation in hidden layers
                if dropout_rate > 0.0:
                    self.layers.append(nnx.Dropout(dropout_rate, rngs=rngs))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Performs the forward pass through the MLP.

        Args:
            x: The input tensor.

        Returns:
            The output tensor after passing through the MLP.
        """
        # Apply periodic transformations if any periodic indices are specified
        if self.periodic_in_features:
            x = self._apply_periodic_transform(x)

        # iteratively apply all layers
        for layer in self.layers:
            x = layer(x)

        # Apply output range scaling if specified
        if self.out_feature_min is not None or self.out_feature_max is not None:
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
            Scaled tensor with values constrained to the specified ranges.
        """
        if self.out_feature_min is None and self.out_feature_max is None:
            return x

        # Handle different cases based on which bounds are provided
        if (
            self.out_feature_min is not None
            and self.out_feature_max is not None
        ):
            # Both min and max are provided - handle per-feature
            min_vals = self.out_feature_min
            max_vals = self.out_feature_max

            # Check for infinite bounds per feature
            has_finite_min = jnp.isfinite(min_vals)
            has_finite_max = jnp.isfinite(max_vals)

            # Case 1: Both bounds are finite - use sigmoid scaling
            both_finite = has_finite_min & has_finite_max
            x_sigmoid = nnx.sigmoid(x)
            finite_scaled = min_vals + (max_vals - min_vals) * x_sigmoid

            # Case 2: Only min is finite (+inf upper bound) - use softplus
            only_min_finite = has_finite_min & ~has_finite_max
            min_scaled = min_vals + nnx.softplus(x)

            # Case 3: Only max is finite (-inf lower bound) - use neg softplus
            only_max_finite = ~has_finite_min & has_finite_max
            max_scaled = max_vals - nnx.softplus(-x)

            # Combine results using where clauses
            result = jnp.where(both_finite, finite_scaled, x)
            result = jnp.where(only_min_finite, min_scaled, result)
            result = jnp.where(only_max_finite, max_scaled, result)

            return result

        # If only min is provided
        if self.out_feature_min is not None:
            min_vals = self.out_feature_min
            has_finite_min = jnp.isfinite(min_vals)

            # Only apply scaling where min is finite
            scaled_x = min_vals + nnx.softplus(x)
            return jnp.where(has_finite_min, scaled_x, x)

        # If only max is provided
        if self.out_feature_max is not None:
            max_vals = self.out_feature_max
            has_finite_max = jnp.isfinite(max_vals)

            # Only apply scaling where max is finite
            scaled_x = max_vals - nnx.softplus(-x)
            return jnp.where(has_finite_max, scaled_x, x)

        return x
