from abc import ABC, abstractmethod

import jax
from flax import nnx, struct


@struct.dataclass
class System(nnx.Module, ABC):
    @abstractmethod
    def step(self, state: jax.Array, action: jax.Array) -> jax.Array:
        pass
