from dataclasses import dataclass

import jax
import jax.numpy as jnp

from hcnn.constraint.constraint import Constraint


@dataclass(frozen=True)
class ConstraintStateOutsideSphere(Constraint):
    center: jax.Array

    def __new__(cls, *args, **kwargs):
        raise TypeError("""
                        Direct instantiation is not allowed.
                        Use `ConstraintStateOutsideSphere.factory()` instead.
                        """)

    @classmethod
    def factory(cls, center: jax.Array, radius: float) -> "ConstraintStateOutsideSphere":
        """
        Factory method to create a ConstraintStateOutsideSphere instance.
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "lower_bound", jnp.array(radius**2))
        object.__setattr__(obj, "upper_bound", jnp.array(jnp.inf))
        object.__setattr__(obj, "center", center)
        return obj

    def inequality(self, state: jax.Array, action: jax.Array) -> jax.Array:
        vector_to_center = state - self.center
        distance_squared = jnp.dot(vector_to_center, vector_to_center)
        return distance_squared
