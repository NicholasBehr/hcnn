from abc import ABC, abstractmethod

import jax
from flax import struct


@struct.dataclass
class SystemCostBase(ABC):
    state_dim: int
    action_dim: int

    @jax.jit
    @abstractmethod
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """Computes the cost given the current state and action.

        Args:
            state (jax.Array): Current state,
                shape (state_dim,) or (batch_dim, state_dim).
            action (jax.Array): Action,
                shape (action_dim,) or (batch_dim, action_dim).

        Returns:
            jax.Array: Cost,
                shape () or (batch_dim,).
        """
        pass
