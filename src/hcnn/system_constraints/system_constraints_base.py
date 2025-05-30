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

    @abstractmethod
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """
        Evaluates the constraint function g(state, action) for the given state and action.

        Returns:
            jax.Array: A 1D array of constraint values of shape (constraint_dim,).

        The constraint is considered satisfied if, for each element i:
            lower_bound[i] <= g(state, action)[i] <= upper_bound[i]

        Args:
            state (jax.Array): The system state of shape (state_dim,).
            action (jax.Array): The control action of shape (action_dim,).
        """
        pass
