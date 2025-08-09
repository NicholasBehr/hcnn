import jax
import jax.numpy as jnp
from flax import nnx, struct


@struct.dataclass
class SamplerUniform(nnx.Module):
    """NNX module for sampling random initial states with specified bounds.

    This module generates batches of random initial states using uniform
    distribution between specified lower and upper bounds for each feature.
    """

    state_dim: int
    batch_size: int
    lower_bound: jax.Array  # Shape (state_dim,)
    upper_bound: jax.Array  # Shape (state_dim,)
    rngs: nnx.Rngs

    @classmethod
    def factory(
        cls,
        state_dim: int,
        batch_size: int,
        lower_bound: jax.Array,
        upper_bound: jax.Array,
        rngs: nnx.Rngs | None = None,
    ) -> "SamplerUniform":
        """Factory method to create a SamplerUniform instance.

        Args:
            state_dim: Dimension of the state space.
            batch_size: Number of initial states to generate per call.
            lower_bound: Lower bounds for each state feature.
                Shape (state_dim,).
            upper_bound: Upper bounds for each state feature.
                Shape (state_dim,).
            rngs: NNX Rngs object for random number generation.

        Returns:
            A SamplerUniform instance.

        Raises:
            ValueError: If bounds have incorrect shapes or
                lower >= upper bounds.
        """
        # Get JAX's default float type to respect 32/64 bit config
        float_dtype = jnp.ones(1).dtype

        # Cast to floating respecting JAX config
        lower_bound = jnp.asarray(lower_bound, dtype=float_dtype)
        upper_bound = jnp.asarray(upper_bound, dtype=float_dtype)

        # Validate shapes
        if lower_bound.shape != (state_dim,):
            raise ValueError(
                f"lower_bound must have shape ({state_dim},), "
                f"got {lower_bound.shape}"
            )
        if upper_bound.shape != (state_dim,):
            raise ValueError(
                f"upper_bound must have shape ({state_dim},), "
                f"got {upper_bound.shape}"
            )

        # Validate bounds order
        if jnp.any(lower_bound >= upper_bound):
            raise ValueError(
                "lower_bound must be strictly less than upper_bound "
                "for all features"
            )

        # Validate batch_size
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # Create rngs if not provided
        if rngs is None:
            rngs = nnx.Rngs(0)  # Use default seed of 0

        return cls(
            state_dim=state_dim,
            batch_size=batch_size,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            rngs=rngs,
        )

    def __call__(self) -> jax.Array:
        """Generate a batch of random initial states.

        Returns:
            A JAX array of shape (batch_size, state_dim) containing random
            initial states sampled uniformly between lower_bound and
            upper_bound.
        """
        # Sample directly between bounds using minval and maxval
        samples = jax.random.uniform(
            key=self.rngs(),
            shape=(self.batch_size, self.state_dim),
            dtype=self.lower_bound.dtype,
            minval=self.lower_bound,
            maxval=self.upper_bound,
        )

        return samples
