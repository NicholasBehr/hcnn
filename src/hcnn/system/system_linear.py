import jax

from .system import System


class SystemLinear(System):
    state_matrix: jax.Array
    input_matrix: jax.Array

    def __init__(self, state_matrix: jax.Array, input_matrix: jax.Array):
        # (1) verify input
        if state_matrix.ndim != 2 or input_matrix.ndim != 2:  # noqa: PLR2004
            raise ValueError("Inout and state matrices must be two dimensional!")
        if state_matrix.shape[0] != state_matrix.shape[1]:
            raise ValueError("Input matrix must be square!")
        if state_matrix.shape[0] != input_matrix.shape[0]:
            raise ValueError("State dimensions must agree!")

        # (2) assign input
        self.state_matrix = state_matrix
        self.input_matrix = input_matrix

    @property
    def state_dim(self) -> int:
        return self.state_matrix.shape[0]

    @property
    def input_dim(self) -> int:
        return self.input_matrix.shape[1]

    def step(self, state: jax.Array, action: jax.Array) -> jax.Array:
        return self.state_matrix @ state + self.input_matrix @ action
