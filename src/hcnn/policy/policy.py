from abc import ABC, abstractmethod

import jax
from flax import struct


@struct.dataclass
class Policy(ABC):
    @abstractmethod
    def get_action(self, state: jax.Array) -> jax.Array:
        pass

    @abstractmethod
    def get_value(self, state: jax.Array) -> jax.Array:
        pass
