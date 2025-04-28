import jax
import jax.numpy as jnp
from flax import struct

from hcnn.optimization.optimization import OptimizationProblem
from hcnn.optimization.solver_linearize import SolverLinearize


@struct.dataclass
class TestProblem(OptimizationProblem):
    @staticmethod
    def factory(x0: jax.Array) -> "TestProblem":
        return TestProblem(
            x0=x0,
            cost_linear=jnp.array([0.0, 0.0]),
            lower_bound=jnp.array([1.0]),
            upper_bound=jnp.array([jnp.inf]),
        )

    ## cost
    def cost_qvp(self, x: jax.Array) -> jax.Array:
        # Q = jnp.eye(2)
        return x

    ## constraints
    def constraints(self, x: jax.Array) -> jax.Array:
        test_constraints = jnp.dot(x, x)
        return jnp.atleast_1d(test_constraints)


def test_solve_start_feasible():
    problem = TestProblem.factory(jnp.array([2.0, 2.0]))
    x = SolverLinearize.solve(problem, verbose=True)

    assert jnp.all(problem.lower_bound - 1e-6 <= problem.constraints(x))
    assert jnp.all(problem.constraints(x) <= problem.upper_bound + 1e-6)


def test_solve_start_infeasible():
    problem = TestProblem.factory(jnp.array([0.5, 0.5]))
    x = SolverLinearize.solve(problem, verbose=True)

    assert jnp.all(problem.lower_bound - 1e-6 <= problem.constraints(x))
    assert jnp.all(problem.constraints(x) <= problem.upper_bound + 1e-6)
