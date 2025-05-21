from functools import partial

import jax
import jax.numpy as jnp
import pytest
from jax import test_util

from hcnn.optimization.solver_osqp import OptimizationSolverOSQP
from tests.optimization.optimization import get_problem_and_phi

TOLERANCE = 1e-5


@pytest.fixture
def solver():
    return OptimizationSolverOSQP(tol=TOLERANCE, max_iter=5000)


def test_solve(solver: OptimizationSolverOSQP):
    for i in range(5):
        rng = jax.random.key(i)
        problem, phi = get_problem_and_phi(rng)
        variables = solver._solve_variables(problem, phi)
        print(f"iter counter: {variables.iter_counter}")
        assert jnp.max(jnp.abs(variables.primal_residuals)) <= TOLERANCE
        assert jnp.max(jnp.abs(variables.dual_residuals)) <= TOLERANCE


def test_vjp(solver: OptimizationSolverOSQP):
    for i in range(5):
        rng = jax.random.key(i)
        problem, phi = get_problem_and_phi(rng)

        print(f"running test {i}")

        test_util.check_vjp(
            f=solver.solve,
            f_vjp=partial(jax.vjp, solver.solve),
            args=(problem, phi),
            eps=10 * TOLERANCE,
            rtol=5 / 100,
            err_msg=f"experiment {i} failed",
        )

        print(f"completing test {i}")
