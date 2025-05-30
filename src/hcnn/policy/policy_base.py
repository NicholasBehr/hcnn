from abc import ABC, abstractmethod

import jax
from flax import nnx, struct


@struct.dataclass
class PolicyBase(nnx.Module, ABC):
    state_dim: int
    action_dim: int

    @abstractmethod
    def __call__(
        self,
        state: jax.Array,
        *,
        jax_key: jax.Array | None = None,
    ) -> jax.Array:
        """
        Compute the action for a given state.

        Args:
            state (jax.Array): Input state vector of shape (state_dim,).
            jax_key (jax.Array, optional): Optional PRNG key for stochastic policies.

        Returns:
            jax.Array: The action vector of shape (action_dim,).
        """
        pass
