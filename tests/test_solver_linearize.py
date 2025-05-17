import jax
import jax.numpy as jnp
from flax import struct

from hcnn.optimization.optimization import OptimizationProblem
from hcnn.optimization.solver_linearize import OptimizationSolverLinearize


@struct.dataclass
class Problem(OptimizationProblem):
    _x0: jax.Array

    @property
    def x0(self) -> jax.Array:
        return self._x0

    @staticmethod
    def factory(x0: jax.Array) -> "Problem":
        return Problem(
            cost_linear=jnp.array([0.0, 0.0]),
            lower_bound=jnp.array([1.0, -jnp.inf]),
            upper_bound=jnp.array([jnp.inf, 2.0]),
            _x0=x0,
        )

    ## cost
    def cost_qvp(self, x: jax.Array) -> jax.Array:
        # Q = jnp.eye(2)
        return x

    ## constraints
    def constraints(self, x: jax.Array) -> jax.Array:
        test_constraints = jnp.dot(x, x)
        return jnp.tile(test_constraints, 2)


def test_solve_start_feasible():
    problem = Problem.factory(jnp.array([2.0, 2.0]))
    x = OptimizationSolverLinearize(num_iter=5).solve(problem)

    assert jnp.all(problem.lower_bound - 1e-6 <= problem.constraints(x))
    assert jnp.all(problem.constraints(x) <= problem.upper_bound + 1e-6)


def test_solve_start_infeasible():
    problem = Problem.factory(jnp.array([0.5, 0.5]))
    x = OptimizationSolverLinearize(num_iter=5).solve(problem)

    assert jnp.all(problem.lower_bound - 1e-6 <= problem.constraints(x))
    assert jnp.all(problem.constraints(x) <= problem.upper_bound + 1e-6)
