from abc import ABC, abstractmethod

import jax
from flax import nnx, struct


@struct.dataclass
class PolicyBase(nnx.Module, ABC):
    state_dim: int = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)

    @abstractmethod
    def __call__(
        self,
        state: jax.Array,
        *,
        jax_key: jax.Array | None = None,
    ) -> jax.Array:
        """Computes the policy action given the current state.

        Args:
            state (jax.Array): Current state, shape (..., state_dim).
            jax_key (jax.Array, optional):
                Optional PRNG key for stochastic policies.

        Returns:
            jax.Array: Policy action, shape (..., action_dim).
        """
        pass
