import jax
from flax import struct
from jax import numpy as jnp

from .system import System


@struct.dataclass
class SystemLinear(System):
    state_matrix: jax.Array
    input_matrix: jax.Array

    def __post_init__(self):
        if self.state_matrix.shape[0] != self.state_matrix.shape[1]:
            raise ValueError("Input matrix must be square!")
        if self.state_matrix.shape[0] != self.input_matrix.shape[0]:
            raise ValueError("State dimensions must agree!")

    @classmethod
    def factory(cls, state_matrix: jax.Array, input_matrix: jax.Array) -> "SystemLinear":
        state_matrix = jnp.atleast_2d(state_matrix)
        input_matrix = jnp.atleast_2d(input_matrix)
        return SystemLinear(
            state_matrix=state_matrix,
            input_matrix=input_matrix,
        )

    def step(self, state: jax.Array, action: jax.Array) -> jax.Array:
        return self.state_matrix @ state + self.input_matrix @ action
