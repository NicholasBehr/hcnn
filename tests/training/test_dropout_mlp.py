import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hcnn.training.dropout_mlp import DropoutMLP


class TestDropoutMLP:
    """Test class for DropoutMLP."""

    @pytest.fixture
    def key(self):
        return jax.random.key(42)

    @pytest.fixture
    def rngs(self, key):
        return nnx.Rngs(key)

    @pytest.fixture
    def in_features(self):
        return 4

    @pytest.fixture
    def out_features(self):
        return 3

    @pytest.fixture
    def hidden_features(self):
        return [8, 6]

    @pytest.fixture
    def basic_mlp(self, in_features, out_features, hidden_features, rngs):
        """Basic MLP without any special features."""
        return DropoutMLP(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            rngs=rngs,
        )

    @pytest.fixture
    def batch_input(self, in_features):
        """Sample batch input for testing."""
        return jnp.ones((5, in_features))

    def test_basic_creation(self, basic_mlp, in_features, out_features):
        """Test basic MLP creation without special features."""
        assert isinstance(basic_mlp, DropoutMLP)
        assert basic_mlp.out_feature_min is None
        assert basic_mlp.out_feature_max is None
        assert len(basic_mlp.periodic_in_features) == 0

    def test_basic_forward_pass(self, basic_mlp, batch_input, out_features):
        """Test basic forward pass."""
        output = basic_mlp(batch_input)
        assert output.shape == (5, out_features)
        assert jnp.isfinite(output).all()

    def test_dropout_creation(
        self, in_features, out_features, hidden_features, rngs
    ):
        """Test MLP creation with dropout."""
        mlp = DropoutMLP(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            rngs=rngs,
            dropout_rate=0.3,
        )
        assert isinstance(mlp, DropoutMLP)
        # Check that dropout layers are included
        dropout_layers = [
            layer for layer in mlp.layers if isinstance(layer, nnx.Dropout)
        ]
        assert len(dropout_layers) == len(hidden_features)

    def test_activation_functions(
        self, in_features, out_features, hidden_features, rngs, batch_input
    ):
        """Test different activation functions."""
        activations = ["relu", "tanh", "leaky_relu", "celu", "hard_sigmoid"]

        for activation in activations:
            mlp = DropoutMLP(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                rngs=rngs,
                activation=activation,
            )
            output = mlp(batch_input)
            assert output.shape == (5, out_features)
            assert jnp.isfinite(output).all()

    def test_periodic_features(self, out_features, hidden_features, rngs):
        """Test periodic feature transformations."""
        in_features = 4
        periodic_indices = frozenset([1, 3])  # Make features 1 and 3 periodic

        mlp = DropoutMLP(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            rngs=rngs,
            periodic_in_features=periodic_indices,
        )

        assert mlp.periodic_in_features == periodic_indices

        # Input with 4 features, 2 of which are periodic
        x = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        output = mlp(x)
        assert output.shape == (1, out_features)
        assert jnp.isfinite(output).all()

    def test_output_range_both_min_max(
        self, in_features, out_features, hidden_features, rngs, batch_input
    ):
        """Test output range scaling with both min and max."""
        min_vals = jnp.array([-2.0, 0.0, 1.0])
        max_vals = jnp.array([2.0, 5.0, 10.0])

        mlp = DropoutMLP(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            rngs=rngs,
            out_feature_min=min_vals,
            out_feature_max=max_vals,
        )

        output = mlp(batch_input)
        assert output.shape == (5, out_features)
        assert jnp.isfinite(output).all()

        # Check that outputs are within specified ranges
        assert jnp.all(output[:, 0] >= min_vals[0])
        assert jnp.all(output[:, 0] <= max_vals[0])
        assert jnp.all(output[:, 1] >= min_vals[1])
        assert jnp.all(output[:, 1] <= max_vals[1])
        assert jnp.all(output[:, 2] >= min_vals[2])
        assert jnp.all(output[:, 2] <= max_vals[2])

    def test_output_range_min_only(
        self, in_features, out_features, hidden_features, rngs, batch_input
    ):
        """Test output range scaling with only minimum values."""
        min_vals = jnp.array([1.0, -5.0, 0.0])

        mlp = DropoutMLP(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            rngs=rngs,
            out_feature_min=min_vals,
        )

        output = mlp(batch_input)
        assert output.shape == (5, out_features)
        assert jnp.isfinite(output).all()

        # Check that outputs are above minimum values
        assert jnp.all(output[:, 0] >= min_vals[0])
        assert jnp.all(output[:, 1] >= min_vals[1])
        assert jnp.all(output[:, 2] >= min_vals[2])

    def test_output_range_max_only(
        self, in_features, out_features, hidden_features, rngs, batch_input
    ):
        """Test output range scaling with only maximum values."""
        max_vals = jnp.array([2.0, 0.0, -1.0])

        mlp = DropoutMLP(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            rngs=rngs,
            out_feature_max=max_vals,
        )

        output = mlp(batch_input)
        assert output.shape == (5, out_features)
        assert jnp.isfinite(output).all()

        # Check that outputs are below maximum values
        assert jnp.all(output[:, 0] <= max_vals[0])
        assert jnp.all(output[:, 1] <= max_vals[1])
        assert jnp.all(output[:, 2] <= max_vals[2])

    def test_combined_features(self, rngs):
        """Test MLP with combined features: dropout, periodic, output ranges."""
        in_features = 6
        out_features = 2
        hidden_features = [8, 4]
        periodic_indices = frozenset([0, 2, 4])  # 3 periodic features
        min_vals = jnp.array([0.0, -1.0])
        max_vals = jnp.array([1.0, 1.0])

        mlp = DropoutMLP(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            rngs=rngs,
            activation="tanh",
            dropout_rate=0.1,
            periodic_in_features=periodic_indices,
            out_feature_min=min_vals,
            out_feature_max=max_vals,
        )

        x = jnp.ones((3, in_features))
        output = mlp(x)
        assert output.shape == (3, out_features)
        assert jnp.isfinite(output).all()

        # Check output constraints
        assert jnp.all(output >= min_vals)
        assert jnp.all(output <= max_vals)

    def test_invalid_dropout_rate(
        self, in_features, out_features, hidden_features, rngs
    ):
        """Test invalid dropout rate validation."""
        with pytest.raises(ValueError, match="dropout_rate must be in range"):
            DropoutMLP(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                rngs=rngs,
                dropout_rate=-0.1,
            )

        with pytest.raises(ValueError, match="dropout_rate must be in range"):
            DropoutMLP(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                rngs=rngs,
                dropout_rate=1.1,
            )

    def test_invalid_output_range_shapes(
        self, in_features, out_features, hidden_features, rngs
    ):
        """Test validation of output range array shapes."""
        # Wrong shape for min
        with pytest.raises(ValueError, match="out_feature_min must have shape"):
            DropoutMLP(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                rngs=rngs,
                out_feature_min=jnp.array([1.0, 2.0]),  # Wrong shape
            )

        # Wrong shape for max
        with pytest.raises(ValueError, match="out_feature_max must have shape"):
            DropoutMLP(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                rngs=rngs,
                out_feature_max=jnp.array([1.0, 2.0, 3.0, 4.0]),  # Wrong shape
            )

    def test_invalid_min_max_relationship(
        self, in_features, out_features, hidden_features, rngs
    ):
        """Test validation that min < max for all features."""
        min_vals = jnp.array([1.0, 2.0, 3.0])
        # min[0] > max[0], min[2] > max[2]
        max_vals = jnp.array([0.0, 3.0, 2.0])

        with pytest.raises(
            ValueError,
            match="out_feature_min must be less than out_feature_max",
        ):
            DropoutMLP(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                rngs=rngs,
                out_feature_min=min_vals,
                out_feature_max=max_vals,
            )

    def test_external_rngs(self, basic_mlp, batch_input, key):
        """Test using external RNGs during forward pass."""
        external_rngs = nnx.Rngs(jax.random.key(123))

        # Should work with external RNGs
        output1 = basic_mlp(batch_input, rngs=external_rngs)

        # Should also work with default RNGs (None)
        output2 = basic_mlp(batch_input, rngs=None)

        # Both should have correct shape
        assert output1.shape == output2.shape
        assert jnp.isfinite(output1).all()
        assert jnp.isfinite(output2).all()

    def test_single_input_vector(self, basic_mlp, in_features, out_features):
        """Test with single input vector (no batch dimension)."""
        x = jnp.ones(in_features)
        output = basic_mlp(x)
        assert output.shape == (out_features,)
        assert jnp.isfinite(output).all()

    def test_different_batch_sizes(self, basic_mlp, in_features, out_features):
        """Test with different batch sizes."""
        for batch_size in [1, 10, 100]:
            x = jnp.ones((batch_size, in_features))
            output = basic_mlp(x)
            assert output.shape == (batch_size, out_features)
            assert jnp.isfinite(output).all()

    def test_no_hidden_layers(
        self, in_features, out_features, rngs, batch_input
    ):
        """Test MLP with no hidden layers (direct linear mapping)."""
        mlp = DropoutMLP(
            in_features=in_features,
            out_features=out_features,
            hidden_features=[],  # No hidden layers
            rngs=rngs,
        )

        output = mlp(batch_input)
        assert output.shape == (5, out_features)
        assert jnp.isfinite(output).all()

        # Should only have one layer (Linear)
        assert len(mlp.layers) == 1
        assert isinstance(mlp.layers[0], nnx.Linear)

    def test_periodic_transform_correctness(self, rngs):
        """Test that periodic transformations are applied correctly."""
        in_features = 3
        out_features = 2
        periodic_indices = frozenset([1])  # Only feature 1 is periodic

        mlp = DropoutMLP(
            in_features=in_features,
            out_features=out_features,
            hidden_features=[4],
            rngs=rngs,
            periodic_in_features=periodic_indices,
        )

        # Test specific input
        x = jnp.array([[1.0, jnp.pi, 2.0]])  # pi as periodic feature

        # The transformed input should have shape (1, 4) instead of (1, 3)
        # because feature 1 becomes 2 features (sin, cos)
        transformed = mlp._apply_periodic_transform(x)
        assert transformed.shape == (1, 4)
        assert jnp.isclose(transformed[0, 1], 0.0, atol=1e-6)  # sin(pi) ≈ 0
        assert jnp.isclose(transformed[0, 2], -1.0, atol=1e-6)  # cos(pi) ≈ -1
