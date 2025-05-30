from abc import ABC, abstractmethod

import jax
from flax import struct


@struct.dataclass
class SystemConstraintsBase(ABC):
    state_dim: int | None
    action_dim: int | None
    constraint_dim: int
    lower_bound: jax.Array  # 1d float array
    upper_bound: jax.Array  # 1d float array

    # def __post_init__(self):
    #     # verify 1d
    #     assert jnp.ndim(self.lower_bound) == 1
    #     assert jnp.ndim(self.upper_bound) == 1

    #     # verify floating
    #     assert jnp.issubdtype(self.lower_bound.dtype, jnp.floating)
    #     assert jnp.issubdtype(self.upper_bound.dtype, jnp.floating)

    #     # verify constraint dim
    #     assert self.lower_bound.shape[0] == self.constraint_dim
    #     assert self.upper_bound.shape[0] == self.constraint_dim

    @abstractmethod
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """
        Returns the constraint function (g(state, action)) evaluated at the given state and action.
        The constraint is satisfied if l <= g(s, a) <= u.
        The lower bound (l), and upper bound (u) are constants.
        """
        pass
