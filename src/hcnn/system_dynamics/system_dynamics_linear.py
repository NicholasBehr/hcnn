import jax
from flax import struct
from jax import numpy as jnp

from hcnn.system_dynamics.system_dynamics_base import SystemDynamicsBase


@struct.dataclass
class SystemDynamicsLinear(SystemDynamicsBase):
    state_matrix_transp: jax.Array  # float 2d matrix
    action_matrix_transp: jax.Array  # float 2d matrix

    @classmethod
    def factory(cls, state_matrix: jax.Array, action_matrix: jax.Array):
        # Get JAX's default float type to respect 32/64 bit config
        float_dtype = jnp.ones(1).dtype

        # Cast to floating respecting JAX config
        state_matrix = jnp.asarray(state_matrix, dtype=float_dtype)
        action_matrix = jnp.asarray(action_matrix, dtype=float_dtype)

        # ensure at least 2d
        state_matrix = jnp.atleast_2d(state_matrix)
        action_matrix = jnp.atleast_2d(action_matrix)

        # verify exactly 2d
        two_dim = 2
        assert jnp.ndim(state_matrix) == two_dim
        assert jnp.ndim(action_matrix) == two_dim

        state_dim = state_matrix.shape[1]
        action_dim = action_matrix.shape[1]

        # verify output dimension
        assert state_matrix.shape == (state_dim, state_dim)
        assert action_matrix.shape == (state_dim, action_dim)

        return SystemDynamicsLinear(
            state_dim=state_dim,
            action_dim=action_dim,
            state_matrix_transp=state_matrix.T,
            action_matrix_transp=action_matrix.T,
        )

    @jax.jit
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        state_dynamics = state @ self.state_matrix_transp
        action_dynamics = action @ self.action_matrix_transp
        return state_dynamics + action_dynamics
