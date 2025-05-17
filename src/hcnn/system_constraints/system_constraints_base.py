from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class SystemConstraintsBase(ABC):
    lower_bound: jax.Array  # 1d array
    upper_bound: jax.Array  # 1d array

    @property
    def constraint_dim(self):
        return self.lower_bound.shape[0]

    def __post_init__(self):
        assert jnp.ndim(self.lower_bound) == 1
        assert jnp.ndim(self.upper_bound) == 1
        assert self.lower_bound.shape[0] == self.upper_bound.shape[0]

    @abstractmethod
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """
        Returns the constraint function (g(state, action)) evaluated at the given state and action.
        The constraint is satisfied if l <= g(s, a) <= u.
        The lower bound (l), and upper bound (u) are constants.
        """
        pass
