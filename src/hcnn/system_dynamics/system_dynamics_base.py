from abc import ABC, abstractmethod

import jax
from flax import struct


@struct.dataclass
class SystemDynamicsBase(ABC):
    # Treat dimension fields as static metadata for JIT shape construction
    state_dim: int = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)

    @abstractmethod
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """Computes the next state given the current state and action.

        Args:
            state (jax.Array): Current state, shape (..., state_dim).
            action (jax.Array): Action, shape (..., action_dim).

        Returns:
            jax.Array: Next state, (..., state_dim).
        """
        pass
