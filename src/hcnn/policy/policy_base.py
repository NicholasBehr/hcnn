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
        pass
