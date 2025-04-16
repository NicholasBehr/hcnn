from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax


@dataclass(frozen=True)
class System(ABC):
    @abstractmethod
    def step(self, state: jax.Array, action: jax.Array) -> jax.Array:
        pass
