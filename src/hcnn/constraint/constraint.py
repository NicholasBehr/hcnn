from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax


@dataclass(frozen=True)
class Constraint(ABC):
    lower_bound: jax.Array
    upper_bound: jax.Array

    @abstractmethod
    def function(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """
        Returns the constraint function (g(state, action)) evaluated at the given state and action.
        The constraint is satisfied if l <= g(s, a) <= u.
        The lower bound (l), and upper bound (u) are constants.
        """
        pass
