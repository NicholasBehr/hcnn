from abc import ABC, abstractmethod

import jax


class Policy(ABC):
    @abstractmethod
    def next_action(self, state: jax.Array) -> jax.Array:
        pass
