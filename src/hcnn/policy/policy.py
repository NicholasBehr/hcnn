from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax


@dataclass(frozen=True)
class Policy(ABC):
    @abstractmethod
    def get_action(self, state: jax.Array) -> jax.Array:
        pass

    @abstractmethod
    def get_value(self, state: jax.Array) -> float:
        pass
