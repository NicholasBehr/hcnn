import jax
from flax import struct
from jax import numpy as jnp

from hcnn.system_dynamics import SystemDynamicsBase


@struct.dataclass
class SystemDynamicsLinear(SystemDynamicsBase):
    state_matrix: jax.Array  # float 2d matrix
    input_matrix: jax.Array  # float 2d matrix

    @classmethod
    def factory(cls, state_matrix: jax.Array, input_matrix: jax.Array):
        # verify floating
        assert jnp.issubdtype(state_matrix.dtype, jnp.floating)
        assert jnp.issubdtype(input_matrix.dtype, jnp.floating)

        state_matrix = jnp.atleast_2d(state_matrix)
        input_matrix = jnp.atleast_2d(input_matrix)

        # verify 2d
        two_dim = 2
        assert jnp.ndim(state_matrix) == two_dim
        assert jnp.ndim(input_matrix) == two_dim

        state_dim = state_matrix.shape[1]
        action_dim = input_matrix.shape[1]

        # verify output dimension
        assert state_matrix.shape == (state_dim, state_dim)
        assert input_matrix.shape == (state_dim, action_dim)

        return SystemDynamicsLinear(
            state_dim=state_dim,
            action_dim=action_dim,
            state_matrix=state_matrix,
            input_matrix=input_matrix,
        )

    def next_state(self, state: jax.Array, action: jax.Array) -> jax.Array:
        return self.state_matrix @ state + self.input_matrix @ action
