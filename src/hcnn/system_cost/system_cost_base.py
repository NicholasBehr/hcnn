from abc import ABC, abstractmethod

import jax
from flax import struct


@struct.dataclass
class SystemCostBase(ABC):
    state_dim: int
    action_dim: int

    @abstractmethod
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """
        Computes the stage cost for a given system state and action.

        Args:
            state (jax.Array): The current state vector of shape (state_dim,).
            action (jax.Array): The action or control input vector of shape (action_dim,).

        Returns:
            jax.Array: Scalar array representing the instantaneous (stage) cost incurred by applying
            the given action in the specified state. The returned array should have shape (1,).

        """
        pass
