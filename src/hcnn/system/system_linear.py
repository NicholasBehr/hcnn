import jax

from .system import System


class SystemLinear(System):
    _matrix_a: jax.Array
    _matrix_b: jax.Array

    def __init__(self, matrix_a: jax.Array, matrix_b: jax.Array):
        if matrix_a.ndim != 2 or matrix_b.ndim != 2:  # noqa: PLR2004
            raise ValueError("Matrices A and B must be two dimensional!")
        if matrix_a.shape[0] != matrix_a.shape[1]:
            raise ValueError("A matrix must be square!")
        if matrix_a.shape[0] != matrix_b.shape[0]:
            raise ValueError("State dimensions of matrix A and matrix B must agree!")

        # assign input
        self._matrix_a = matrix_a
        self._matrix_b = matrix_b

    def step(self, state: jax.Array, action: jax.Array) -> jax.Array:
        return self._matrix_a @ state + self._matrix_b @ action
