from abc import ABC, abstractmethod

import jax
from flax import struct


@struct.dataclass
class SystemDynamicsBase(ABC):
    state_dim: int
    action_dim: int

    @abstractmethod
    def next_state(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """
        Computes the next state of the system given the current state and action.

        Args:
            state (jax.Array): The current state vector of shape (state_dim,).
            action (jax.Array): The action or control input vector of shape (action_dim,).

        Returns:
            jax.Array: The next state vector of shape (state_dim,) after applying the action.
        """
        pass
