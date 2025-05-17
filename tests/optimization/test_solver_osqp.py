import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax import test_util

from hcnn.optimization.optimization import OptimizationProblem
from hcnn.optimization.solver_osqp import OptimizationSolverOSQP


class Module(nnx.Module):
    def __init__(self, a, b):
        self.a = nnx.Param(a)
        self.b = nnx.Variable(b)


def random_problem(rng) -> OptimizationProblem:
    keys = jax.random.split(rng, 5)

    module = Module(
        jax.random.uniform(keys[3], 1, minval=0.6, maxval=2.0),
        jax.random.uniform(keys[4], 1, minval=0.6, maxval=2.0),
    )

    graph, param, other = nnx.split(module, nnx.Param, ...)

    def constraint_fn(x, constraint_params):
        module = nnx.merge(graph, constraint_params, other)
        radius_sqr = jnp.dot(x, x) * module.a * module.b
        return jnp.tile(radius_sqr, 2)

    return OptimizationProblem(
        x0=jax.random.normal(keys[0], 3),
        cost_quad_diag=jnp.ones(3),
        cost_linear=jnp.ones(3),
        lower_bound=jax.random.uniform(keys[1], 2, minval=8, maxval=10),
        upper_bound=jax.random.uniform(keys[2], 2, minval=10, maxval=13),
        constraint_params=param,
        constraint_fn=constraint_fn,
    )


TOLERANCE = 1e-5


@pytest.fixture
def solver():
    return OptimizationSolverOSQP(tol=TOLERANCE, max_iter=1e4)


def test_solve(solver: OptimizationSolverOSQP):
    for i in range(25):
        rng = jax.random.PRNGKey(i)
        problem = random_problem(rng)
        variables = solver._solve_full(problem)
        print(f"iter counter: {variables.iter_counter}")
        assert jnp.max(jnp.abs(variables.primal_residuals)) <= TOLERANCE
        assert jnp.max(jnp.abs(variables.dual_residuals)) <= TOLERANCE


def test_grad(solver: OptimizationSolverOSQP):
    for i in range(25):
        rng = jax.random.PRNGKey(i)
        problem = random_problem(rng)

        test_util.check_vjp(
            f=solver.solve,
            f_vjp=lambda problem: jax.vjp(solver.solve, problem),
            args=(problem,),
            eps=5 * TOLERANCE,
            rtol=5 / 100,
            err_msg=f"experiment {i} failed",
        )
