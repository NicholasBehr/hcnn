import jax
from flax import struct
from jax import numpy as jnp

from hcnn.system_cost.system_cost_base import SystemCostBase


@struct.dataclass
class SystemCostQuadratic(SystemCostBase):
    state_cost_matrix: jax.Array  # float 2d matrix
    action_cost_matrix: jax.Array  # float 2d matrix

    @classmethod
    def factory(
        cls, state_cost_matrix: jax.Array, action_cost_matrix: jax.Array
    ):
        # verify floating
        assert jnp.issubdtype(state_cost_matrix.dtype, jnp.floating)
        assert jnp.issubdtype(action_cost_matrix.dtype, jnp.floating)

        state_cost_matrix = jnp.atleast_2d(state_cost_matrix)
        action_cost_matrix = jnp.atleast_2d(action_cost_matrix)

        # verify 2d
        two_dim = 2
        assert jnp.ndim(state_cost_matrix) == two_dim
        assert jnp.ndim(action_cost_matrix) == two_dim

        state_dim = state_cost_matrix.shape[1]
        action_dim = action_cost_matrix.shape[1]

        # verify square
        assert state_cost_matrix.shape == (state_dim, state_dim)
        assert action_cost_matrix.shape == (action_dim, action_dim)

        return SystemCostQuadratic(
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
