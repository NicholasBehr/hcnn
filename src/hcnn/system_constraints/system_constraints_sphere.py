import jax
import jax.numpy as jnp
from flax import struct

from hcnn.system_constraints.system_constraints_base import (
    SystemConstraintsBase,
)


@struct.dataclass
class SystemConstraintsSphere(SystemConstraintsBase):
    center: jax.Array

    @classmethod
    def factory(
        cls, center: jax.Array, radius: jax.Array
    ) -> "SystemConstraintsSphere":
        # Get JAX's default float type to respect 32/64 bit config
        float_dtype = jnp.ones(1).dtype

        # Cast to floating respecting JAX config
        center = jnp.asarray(center, dtype=float_dtype)
        radius = jnp.asarray(radius, dtype=float_dtype)

        # Ensure the radius is a scalar and center is a 1D array
        assert radius.ndim == 0
        center = jnp.atleast_1d(center)
        assert center.ndim == 1

        return SystemConstraintsSphere(
            state_dim=center.shape[0],
            action_dim=None,
            constraint_dim=1,
            lower_bound=jnp.atleast_1d(radius**2),
            upper_bound=jnp.atleast_1d(jnp.inf),
            center=center,
        )

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        vector_to_center = state - self.center
        distance_squared = jnp.sum(vector_to_center**2, axis=-1, keepdims=True)
        return distance_squared
