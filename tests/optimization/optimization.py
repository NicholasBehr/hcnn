import jax
import jax.numpy as jnp
from flax import struct

from hcnn.optimization.optimization import OptimizationProblemBase

# constaint dimension
p = 20


@struct.dataclass
class ConvexProblem(OptimizationProblemBase[jax.Array]):
    def get_x0(self, phi: jax.Array):
        return jnp.zeros((2 * p,))

    def constraint_fn(self, x: jax.Array, phi: jax.Array) -> jax.Array:
        return jnp.hstack([phi, -jnp.eye(p)]) @ x

    @classmethod
    def factory(cls, jax_key):
        cost_quad_diag = jnp.ones(2 * p)
        cost_linear = jnp.zeros(2 * p)
        phi = jax.random.normal(jax.random.fold_in(jax_key, 1), (p, p))
        lower_bound = -jnp.inf * jnp.ones((p,))
        upper_bound = jax.random.uniform(jax.random.fold_in(jax_key, 2), (p,), minval=-10, maxval=10)

        return ConvexProblem(
            cost_quad_diag=cost_quad_diag,
            cost_linear=cost_linear,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        ), phi


def get_problem_and_phi(jax_key):
    return ConvexProblem.factory(jax_key)
