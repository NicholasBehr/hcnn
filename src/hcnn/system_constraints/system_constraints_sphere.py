import jax
import jax.numpy as jnp
from flax import struct

from hcnn.system_constraints import SystemConstraintsBase


@struct.dataclass
class SystemConstraintsSphere(SystemConstraintsBase):
    center: jax.Array

    @classmethod
    def factory(cls, center: jax.Array, radius: jax.Array) -> "SystemConstraintsSphere":
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
        distance_squared = jnp.dot(vector_to_center, vector_to_center)
        return jnp.atleast_1d(distance_squared)
