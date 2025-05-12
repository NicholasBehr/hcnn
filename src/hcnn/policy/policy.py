from abc import ABC, abstractmethod

import jax
from flax import nnx


class Policy(nnx.Module, ABC):
    @abstractmethod
    def __call__(self, state: jax.Array) -> jax.Array:
        pass
