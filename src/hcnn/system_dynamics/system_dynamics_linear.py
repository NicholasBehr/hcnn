import jax
from flax import struct
from jax import numpy as jnp

from hcnn.system_dynamics import SystemDynamicsBase


@struct.dataclass
class SystemDynamicsLinear(SystemDynamicsBase):
    state_matrix: jax.Array  # float 2d matrix
    input_matrix: jax.Array  # float 2d matrix

    # def __post_init__(self):
    #     # verify floating
    #     assert jnp.issubdtype(self.state_matrix.dtype, jnp.floating)
    #     assert jnp.issubdtype(self.input_matrix.dtype, jnp.floating)

    #     # verify output dimension
    #     assert self.state_matrix.shape == (self.state_dim, self.state_dim)
    #     assert self.input_matrix.shape == (self.state_dim, self.action_dim)

    @classmethod
    def factory(cls, state_matrix: jax.Array, input_matrix: jax.Array):
        state_matrix = jnp.atleast_2d(state_matrix)
        input_matrix = jnp.atleast_2d(input_matrix)
        return SystemDynamicsLinear(
            state_dim=state_matrix.shape[1],
            action_dim=input_matrix.shape[1],
            state_matrix=state_matrix,
            input_matrix=input_matrix,
        )

    def next_state(self, state: jax.Array, action: jax.Array) -> jax.Array:
        return self.state_matrix @ state + self.input_matrix @ action
