from abc import ABC, abstractmethod

import jax
from flax import struct


@struct.dataclass
class SystemConstraintsBase(ABC):
    # Treat dimension fields as static metadata (not pytree leaves)
    state_dim: int = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    constraint_dim: int = struct.field(pytree_node=False)
    lower_bound: jax.Array  # 1d float array
    upper_bound: jax.Array  # 1d float array

    @abstractmethod
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """
        Evaluates the constraint function g(state, action)
        for the given state and action.

        Args:
            state (jax.Array): State, shape (..., state_dim).
            action (jax.Array): Action, shape (..., action_dim).

        Returns:
            jax.Array: Constraint values, shape (..., constraint_dim).

        The constraint is considered satisfied if, for each element i:
            lower_bound[i] <= g(state, action)[i] <= upper_bound[i]
        """
        pass
