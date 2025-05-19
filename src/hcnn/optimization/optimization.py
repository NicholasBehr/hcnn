from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple

import jax
from flax import nnx, struct


class OptimizationProblem(nnx.Module, ABC):
    """
    Optimization Problem Definition
    """

    ### INITIALIZATION ###
    # initial guess for optimization variable x (vectorized 1d)
    @property
    @abstractmethod
    def x0(self) -> jax.Array:
        pass

    ### QUADRATIC COST ###
    # f(x) = 1/2 x.T @ diag(q) @ x + c.T @ x
    cost_quad_diag: nnx.Variable  # q in cost (vectorized 1d)
    cost_linear: nnx.Variable  # c in cost (vectorized 1d)

    ### NONLINEAR PARAMETRIC CONSTRAINTS ###
    # l <= h(x, phi) <= u
    @abstractmethod
    def constraint_fn(self, x: jax.Array) -> jax.Array:
        pass

    lower_bound: jax.Array  # l in constraint (vectorized 1d)
    upper_bound: jax.Array  # u in constraint (vectorized 1d)


@struct.dataclass
class OptimizationSolverBase(ABC):
    """
    Quadratic Cost with Nonlinear Constraints Problem Solver
    """

    def solve(self, problem: OptimizationProblem) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Solve the given optimization problem and return the optimal solution.

        This method takes an `OptimizationProblem` instance containing:
        - a quadratic cost function
        - nonlinear parametric constraints
        - an initial guess for the optimal decision variable `x*`

        The solver implementation should compute and return the optimal
        value of the decision variable that minimizes the quadratic cost
        while satisfying all constraints.

        Parameters:
            problem (OptimizationProblem): The optimization problem to be solved,
                including cost function parameters, constraint bounds, and
                constraint function.

        Returns:
            jax.Array: The optimal solution `x*` for the optimization problem,
                matching the shape of the initial guess `problem.x_0`.
        """

        # argument "self" needs to be passed explicitly for custom vjp to work
        return self._solve_vjp(self, problem)

    @partial(nnx.custom_vjp, nondiff_argnums=(0,))
    @abstractmethod
    def _solve_vjp(self, problem: OptimizationProblem) -> jax.Array:
        pass
