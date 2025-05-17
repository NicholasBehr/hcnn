import jax
import jax.numpy as jnp
from flax import struct

from .system_constraints_base import SystemConstraintsBase


@struct.dataclass
class SystemConstraintsSphere(SystemConstraintsBase):
    center: jax.Array

    @classmethod
    def factory(cls, center: jax.Array, radius: float) -> "SystemConstraintsSphere":
        return SystemConstraintsSphere(
            lower_bound=jnp.atleast_1d(radius**2),
            upper_bound=jnp.atleast_1d(jnp.inf),
            center=jnp.atleast_1d(center),
        )

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        vector_to_center = state - self.center
        distance_squared = jnp.dot(vector_to_center, vector_to_center)
        return jnp.atleast_1d(distance_squared)
