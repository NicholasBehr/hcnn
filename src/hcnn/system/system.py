from abc import ABC, abstractmethod

import jax


class System(ABC):
    @abstractmethod
    def step(self, state: jax.Array, action: jax.Array) -> jax.Array:
        pass
