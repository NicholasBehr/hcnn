import jax
import jax.numpy as jnp
import pytest
from flax import struct

from hcnn.optimization import OptimizationProblemBase, OptimizationSolverOSQP
from tests.optimization.optimization_base import *


@pytest.fixture
def solver():
    return OptimizationSolverOSQP(tol=1e-5, max_iter=1000)


@struct.dataclass
class OSQPProblem(OptimizationProblemBase[jax.Array]):
    def get_x0(self, phi: jax.Array):
        return jnp.zeros_like(self.cost_linear)

    def constraint_fn(self, x: jax.Array, phi: jax.Array) -> jax.Array:
        return phi @ x

    @classmethod
    def factory(cls, dimension, jax_key):
        keys = jax.random.split(jax_key, 5)
        cost_quad_diag = jax.random.uniform(keys[0], (dimension,), minval=0.5, maxval=2.0)
        cost_linear = jax.random.normal(keys[1], (dimension,))
        lower_bound = jax.random.uniform(keys[2], (dimension - 1,), minval=-3, maxval=2)
        upper_bound = jax.random.uniform(keys[3], (dimension - 1,), minval=lower_bound, maxval=4)
        phi = jax.random.uniform(keys[4], (dimension - 1, dimension))

        return OSQPProblem(
            cost_quad_diag=cost_quad_diag,
            cost_linear=cost_linear,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        ), phi
