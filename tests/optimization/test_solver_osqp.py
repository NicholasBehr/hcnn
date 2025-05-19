import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax import test_util

from hcnn.optimization.optimization import OptimizationProblem
from hcnn.optimization.solver_osqp import OptimizationSolverOSQP


class TestProblem(OptimizationProblem):
    def __init__(self, jax_key):
        keys = jax.random.split(jax_key, 5)

        self._x0 = nnx.Variable(jax.random.normal(keys[0], 3))
        self.a = nnx.Param(jax.random.uniform(keys[1], 1, minval=0.6, maxval=2.0))
        self.b = nnx.Variable(jax.random.uniform(keys[2], 1, minval=0.6, maxval=2.0))

        self.cost_quad_diag = nnx.Variable(jnp.ones(3))
        self.cost_linear = nnx.Variable(jnp.ones(3))
        self.lower_bound = nnx.Variable(jax.random.uniform(keys[3], 2, minval=8, maxval=10))
        self.upper_bound = nnx.Variable(jax.random.uniform(keys[4], 2, minval=10, maxval=13))

    @property
    def x0(self):
        return self._x0

    def constraint_fn(self, x):
        radius_sqr = jnp.dot(x, x) * self.a * self.b
        return jnp.tile(radius_sqr, 2)


TOLERANCE = 1e-5


@pytest.fixture
def solver():
    return OptimizationSolverOSQP(tol=TOLERANCE, max_iter=1e4)


def test_solve(solver: OptimizationSolverOSQP):
    for i in range(25):
        rng = jax.random.PRNGKey(i)
        problem = TestProblem(rng)
        variables = solver._solve_variables(problem)
        print(f"iter counter: {variables.iter_counter}")
        assert jnp.max(jnp.abs(variables.primal_residuals)) <= TOLERANCE
        assert jnp.max(jnp.abs(variables.dual_residuals)) <= TOLERANCE


def test_grad(solver: OptimizationSolverOSQP):
    def loss(problem):
        x = solver.solve(problem)
        return jnp.dot(x, x)

    rng = jax.random.PRNGKey(0)
    problem = TestProblem(rng)

    grad = nnx.grad(loss)(problem)

    for i in range(25):
        test_util.check_vjp(
            f=solver.solve,
            f_vjp=lambda problem: jax.vjp(solver.solve, problem),
            args=(problem,),
            eps=5 * TOLERANCE,
            rtol=5 / 100,
            err_msg=f"experiment {i} failed",
        )
