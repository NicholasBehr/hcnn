import jax
import jax.numpy as jnp
from flax import struct

from hcnn.optimization.optimization import OptimizationProblemBase


@struct.dataclass
class Context:
    a: jax.Array
    b: jax.Array
    jax_key: jax.Array = struct.field(pytree_node=False)


@struct.dataclass
class TestProblem(OptimizationProblemBase[Context]):
    def get_x0(self, phi: Context):
        return jax.random.normal(jax.random.fold_in(phi.jax_key, 0), (3,))

    def constraint_fn(self, x: jax.Array, phi: Context) -> jax.Array:
        radius_sqr = jnp.dot(x, x) * phi.a * phi.b
        return jnp.tile(radius_sqr, 5)

    @classmethod
    def factory(cls, jax_key):
        key_a = jax.random.fold_in(jax_key, 1)
        a = jax.random.uniform(key_a, (1,), minval=0.9, maxval=1.1)

        key_b = jax.random.fold_in(jax_key, 2)
        b = jax.random.uniform(key_b, (1,), minval=0.9, maxval=1.1)

        cost_quad_diag = jnp.ones(3)
        cost_linear = jnp.ones(3)
        key_l = jax.random.fold_in(jax_key, 3)
        lower_bound = jax.random.uniform(key_l, (5,), minval=3, maxval=4)
        key_u = jax.random.fold_in(jax_key, 4)
        upper_bound = jax.random.uniform(key_u, (5,), minval=6, maxval=7)

        return TestProblem(
            cost_quad_diag=cost_quad_diag,
            cost_linear=cost_linear,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        ), Context(a, b, jax_key)


def get_problem_and_phi(jax_key):
    return TestProblem.factory(jax_key)
