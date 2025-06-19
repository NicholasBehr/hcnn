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

    @jax.jit
    @abstractmethod
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """
        Evaluates the constraint function g(state, action)
        for the given state and action.

        Args:
            state (jax.Array): State,
                shape (state_dim,) or (batch_size, state_dim).
            action (jax.Array): Action,
                shape (action_dim,) or (batch_size, action_dim).

        Returns:
            jax.Array: Constraint values,
                shape (constraint_dim,) or (batch_size, constraint_dim).

        The constraint is considered satisfied if, for each element i:
            lower_bound[i] <= g(state, action)[i] <= upper_bound[i]
        """
        pass
