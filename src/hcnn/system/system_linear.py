from dataclasses import dataclass

import jax

from .system import System


@dataclass(frozen=True)
class SystemLinear(System):
    state_matrix: jax.Array
    input_matrix: jax.Array

    def __post_init__(self):
        if self.state_matrix.ndim != 2:  # noqa: PLR2004
            raise ValueError("State matrix must be two dimensional!")
        if self.input_matrix.ndim != 2:  # noqa: PLR2004
            raise ValueError("Input matrix must be two dimensional!")
        if self.state_matrix.shape[0] != self.state_matrix.shape[1]:
            raise ValueError("Input matrix must be square!")
        if self.state_matrix.shape[0] != self.input_matrix.shape[0]:
            raise ValueError("State dimensions must agree!")

    @property
    def state_dim(self) -> int:
        return self.state_matrix.shape[0]

    @property
    def input_dim(self) -> int:
        return self.input_matrix.shape[1]

    def step(self, state: jax.Array, action: jax.Array) -> jax.Array:
        return self.state_matrix @ state + self.input_matrix @ action
