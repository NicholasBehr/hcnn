from abc import ABC, abstractmethod

import jax
from flax import struct


@struct.dataclass
class System(ABC):
    @property
    @abstractmethod
    def state_dim(self):
        pass

    @property
    @abstractmethod
    def action_dim(self):
        pass

    @abstractmethod
    def step(self, state: jax.Array, action: jax.Array) -> jax.Array:
        pass
