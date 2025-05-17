import jax
from flax import struct
from jax import numpy as jnp

from .system_dynamics_base import SystemDynamicsBase


@struct.dataclass
class SystemDynamicsLinear(SystemDynamicsBase):
    state_matrix: jax.Array  # 2d array
    input_matrix: jax.Array  # 2d array

    @classmethod
    def factory(cls, state_matrix: jax.Array, input_matrix: jax.Array) -> "SystemDynamicsLinear":
        state_matrix = jnp.atleast_2d(state_matrix)
        input_matrix = jnp.atleast_2d(input_matrix)

        if not jnp.issubdtype(state_matrix.dtype, jnp.floating):
            raise ValueError("State matrix must be of type float!")
        if not jnp.issubdtype(input_matrix.dtype, jnp.floating):
            raise ValueError("Input matrix must be of type float!")
        if state_matrix.shape[0] != state_matrix.shape[1]:
            raise ValueError("Input matrix must be square!")
        if state_matrix.shape[0] != input_matrix.shape[0]:
            raise ValueError("State dimensions must agree!")

        return SystemDynamicsLinear(
            state_dim=state_matrix.shape[1],
            action_dim=input_matrix.shape[1],
            state_matrix=state_matrix,
            input_matrix=input_matrix,
        )

    def step(self, state: jax.Array, action: jax.Array) -> jax.Array:
        return self.state_matrix @ state + self.input_matrix @ action
