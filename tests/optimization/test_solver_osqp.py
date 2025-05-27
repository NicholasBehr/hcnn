import jax
import jax.numpy as jnp
import osqp
import pytest
from flax import struct
from scipy.sparse import csc_matrix

from hcnn.optimization.solver_osqp import OptimizationProblemBase, OptimizationSolverOSQP
from tests.optimization.optimization import *


@pytest.fixture
def solver():
    return OptimizationSolverOSQP(tol=1e-5, max_iter=1_000)


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


def test_primal_dual(solver: OptimizationSolverOSQP):
    dimension = 15
    for i in range(15):
        key = jax.random.key(i)
        problem, phi = OSQPProblem.factory(dimension=dimension, jax_key=key)

        # test subject
        variables = solver._solve_variables(problem, phi)
        x_star = variables.x
        y_star = variables.y

        # reference
        m = osqp.OSQP()
        m.setup(
            P=csc_matrix(jnp.diag(problem.cost_quad_diag)),
            q=problem.cost_linear.__array__(),
            A=csc_matrix(phi),
            l=problem.lower_bound.__array__(),
            u=problem.upper_bound.__array__(),
            eps_rel=1e-5,
            eps_abs=1e-5,
        )
        results = m.solve()
        x_star_ref = results.x
        y_star_ref = results.y

        assert jnp.allclose(x_star, x_star_ref, atol=1e-4)
        assert jnp.allclose(y_star, y_star_ref, atol=1e-4)
