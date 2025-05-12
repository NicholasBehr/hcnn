import jax
import jax.numpy as jnp
from flax import struct

from hcnn.state_action_constraint.state_action_constraint import StateActionConstraint


@struct.dataclass
class StateOutsideSphere(StateActionConstraint):
    center: jax.Array

    @classmethod
    def factory(cls, center: jax.Array, radius: float) -> "StateOutsideSphere":
        return StateOutsideSphere(
            lower_bound=jnp.atleast_1d(radius**2),
            upper_bound=jnp.atleast_1d(jnp.inf),
            center=jnp.atleast_1d(center),
        )

    def function(self, state: jax.Array, action: jax.Array) -> jax.Array:
        vector_to_center = state - self.center
        distance_squared = jnp.dot(vector_to_center, vector_to_center)
        return distance_squared
