from abc import ABC, abstractmethod

import jax
from flax import struct


@struct.dataclass
class StateActionConstraint(ABC):
    lower_bound: jax.Array  # 1d array
    upper_bound: jax.Array  # 1d array

    @abstractmethod
    def function(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """
        Returns the constraint function (g(state, action)) evaluated at the given state and action.
        The constraint is satisfied if l <= g(s, a) <= u.
        The lower bound (l), and upper bound (u) are constants.
        """
        pass
