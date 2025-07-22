from abc import ABC, abstractmethod

import jax
from flax import struct


@struct.dataclass
class SystemCostsBase(ABC):
    state_dim: int
    action_dim: int

    @abstractmethod
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """Computes the cost given the current state and action.

        Args:
            state (jax.Array): Current state, shape (..., state_dim).
            action (jax.Array): Current action, shape (..., action_dim).

        Returns:
            jax.Array: Stage cost, shape (...,).
        """
        pass
