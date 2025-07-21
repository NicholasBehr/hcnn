import jax
from flax import struct
from jax import numpy as jnp

from hcnn.system_costs.system_costs_base import SystemCostsBase


@struct.dataclass
class SystemCostsQuadratic(SystemCostsBase):
    state_cost_matrix: jax.Array  # float 2d matrix
    action_cost_matrix: jax.Array  # float 2d matrix

    @classmethod
    def factory(
        cls, state_cost_matrix: jax.Array, action_cost_matrix: jax.Array
    ):
        # Get JAX's default float type to respect 32/64 bit config
        float_dtype = jnp.ones(1).dtype

        # Cast to floating respecting JAX config
        state_cost_matrix = jnp.asarray(state_cost_matrix, dtype=float_dtype)
        action_cost_matrix = jnp.asarray(action_cost_matrix, dtype=float_dtype)

        # ensure 2d
        state_cost_matrix = jnp.atleast_2d(state_cost_matrix)
        action_cost_matrix = jnp.atleast_2d(action_cost_matrix)

        # verify exacly 2d
        two_dim = 2
        assert jnp.ndim(state_cost_matrix) == two_dim
        assert jnp.ndim(action_cost_matrix) == two_dim

        state_dim = state_cost_matrix.shape[1]
        action_dim = action_cost_matrix.shape[1]

        # verify square
        assert state_cost_matrix.shape == (state_dim, state_dim)
        assert action_cost_matrix.shape == (action_dim, action_dim)

        return SystemCostsQuadratic(
            state_dim=state_dim,
            action_dim=action_dim,
            state_cost_matrix=state_cost_matrix,
            action_cost_matrix=action_cost_matrix,
        )

    @jax.jit
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        state_cost = jnp.sum(
            (state @ self.state_cost_matrix) * state,
            axis=-1,
        )
        action_cost = jnp.sum(
            (action @ self.action_cost_matrix) * action,
            axis=-1,
        )
        return state_cost + action_cost
