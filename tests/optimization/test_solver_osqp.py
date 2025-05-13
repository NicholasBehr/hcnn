import jax
import jax.numpy as jnp
import pytest
from jax import test_util

from hcnn.optimization.optimization import OptimizationProblem
from hcnn.optimization.solver_osqp import SolverOSQP


def random_problem(rng) -> OptimizationProblem:
    def constraint_fn(x, constraint_params):
        radius_sqr = jnp.dot(x, x) * constraint_params
        return jnp.tile(radius_sqr, 2)

    keys = jax.random.split(rng, 4)

    return OptimizationProblem(
        x0=jax.random.normal(keys[0], 3),
        cost_quad_diag=jnp.ones(3),
        cost_linear=jnp.ones(3),
        lower_bound=jax.random.uniform(keys[1], 2, minval=3, maxval=10),
        upper_bound=jax.random.uniform(keys[2], 2, minval=10, maxval=17),
        constraint_params=jax.random.uniform(keys[3], 1, minval=0.2, maxval=2.0),
        constraint_fn=constraint_fn,
    )


TOLERANCE = 1e-5


@pytest.fixture
def solver():
    return SolverOSQP(tol=TOLERANCE)


def test_solve(solver: SolverOSQP):
    for i in range(25):
        rng = jax.random.PRNGKey(i)
        problem = random_problem(rng)
        variables = solver._solve_full(problem)
        print(f"iter counter: {variables.iter_counter}")
        assert jnp.max(jnp.abs(variables.primal_residuals)) <= TOLERANCE
        assert jnp.max(jnp.abs(variables.dual_residuals)) <= TOLERANCE


def test_grad(solver: SolverOSQP):
    for i in range(25):
        rng = jax.random.PRNGKey(i)
        problem = random_problem(rng)

        test_util.check_vjp(
            f=solver.solve,
            f_vjp=lambda problem: jax.vjp(solver.solve, problem),
            args=(problem,),
            eps=5 * TOLERANCE,
            rtol=1 / 100,
            err_msg=f"experiment {i} failed",
        )
