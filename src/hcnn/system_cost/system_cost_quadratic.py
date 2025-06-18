import jax
from flax import struct
from jax import numpy as jnp

from hcnn.system_cost import SystemCostBase


@struct.dataclass
class SystemCostQuadratic(SystemCostBase):
    state_cost: jax.Array  # float 2d matrix
    action_cost: jax.Array  # float 2d matrix

    @classmethod
    def factory(cls, state_cost: jax.Array, action_cost: jax.Array):
        # verify floating
        assert jnp.issubdtype(state_cost.dtype, jnp.floating)
        assert jnp.issubdtype(action_cost.dtype, jnp.floating)

        state_cost = jnp.atleast_2d(state_cost)
        action_cost = jnp.atleast_2d(action_cost)

        # verify 2d
        two_dim = 2
        assert jnp.ndim(state_cost) == two_dim
        assert jnp.ndim(action_cost) == two_dim

        state_dim = state_cost.shape[1]
        action_dim = action_cost.shape[1]

        # verify square
        assert state_cost.shape == (state_dim, state_dim)
        assert action_cost.shape == (action_dim, action_dim)

        return SystemCostQuadratic(
            state_dim=state_dim,
            action_dim=action_dim,
            state_cost=state_cost,
            action_cost=action_cost,
        )

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        a = jnp.sum((state @ self.state_cost) * state, axis=-1)
        b = jnp.sum((action @ self.action_cost) * action, axis=-1)
        return a + b
