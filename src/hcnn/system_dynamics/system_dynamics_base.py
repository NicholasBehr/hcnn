from abc import ABC, abstractmethod

import jax
from flax import struct


@struct.dataclass
class SystemDynamicsBase(ABC):
    state_dim: int
    action_dim: int

    @abstractmethod
    def step(self, state: jax.Array, action: jax.Array) -> jax.Array:
        pass
