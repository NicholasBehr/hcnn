import jax
from flax import struct
from jax import numpy as jnp

from hcnn.system_dynamics.system_dynamics_base import SystemDynamicsBase


@struct.dataclass
class SystemDynamicsLinear(SystemDynamicsBase):
    state_matrix_transp: jax.Array  # float 2d matrix
    input_matrix_transp: jax.Array  # float 2d matrix

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
            state_matrix_transp=state_matrix.T,
            input_matrix_transp=input_matrix.T,
        )

    @jax.jit
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        state_dynamics = state @ self.state_matrix_transp
        input_dynamics = action @ self.input_matrix_transp
        return state_dynamics + input_dynamics
