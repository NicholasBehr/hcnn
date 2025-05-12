import jax
import jax.numpy as jnp
import pytest
from jax import test_util

import hcnn.optimization.solver_osqp as osqp
from hcnn.optimization.optimization import OptimizationProblem


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
        constraint_params=jax.random.uniform(keys[3], 1, minval=0.2, maxval=2),
        constraint_fn=constraint_fn,
    )


TOLERANCE = 1e-6


@pytest.fixture
def solver():
    return osqp.SolverOSQP(tol=TOLERANCE)


def test_solve(solver: osqp.SolverOSQP):
    for i in range(10):
        rng = jax.random.PRNGKey(i)
        problem = random_problem(rng)
        variables = solver._solve_internal(problem)
        print(f"iter counter: {variables.iter_counter}")
        assert jnp.max(jnp.abs(variables.primal_residuals)) <= TOLERANCE
        assert jnp.max(jnp.abs(variables.dual_residuals)) <= TOLERANCE


def test_grad(solver: osqp.SolverOSQP):
    for i in range(10):
        rng = jax.random.PRNGKey(i)
        problem = random_problem(rng)

        test_util.check_vjp(
            f=lambda problem: osqp._solve_custom_vjp_wrapper(solver, problem),
            f_vjp=lambda problem: jax.vjp(
                lambda p: osqp._solve_custom_vjp_wrapper(solver, p), problem
            ),
            args=(problem,),
            eps=5 * TOLERANCE,
            rtol=5 / 100,
            err_msg=f"experiment {i} failed",
        )
