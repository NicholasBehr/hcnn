from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import jax
from flax import struct


@struct.dataclass
class OptimizationProblem:
    """
    Optimization Problem Definition
    """

    x0: jax.Array  # initial guess for optimization variable (vectorized 1d)

    ### QUADRATIC COST ###
    # f(x) = 1/2 x.T @ diag(q) @ x + c.T @ x
    cost_quad_diag: jax.Array  # q in cost (vectorized 1d)
    cost_linear: jax.Array  # c in cost (vectorized 1d)

    ### NONLINEAR PARAMETRIC CONSTRAINTS ###
    # l <= h(x, phi) <= u
    lower_bound: jax.Array  # l in constraint (vectorized 1d)
    upper_bound: jax.Array  # u in constraint (vectorized 1d)
    constraint_params: jax.Array  # phi in constraint (vectorized 1d)
    constraint_fn: Callable[[jax.Array, jax.Array], jax.Array] = struct.field(
        pytree_node=False
    )  # h in constraint definition

    def constraint_fn_x(self, x: jax.Array) -> jax.Array:
        return self.constraint_fn(x, self.constraint_params)


@struct.dataclass
class OptimizationSolver(ABC):
    """
    Quadratic Cost with Nonlinear Constraints Problem Solver
    """

    def solve(self, problem: OptimizationProblem) -> jax.Array:
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

    @partial(jax.custom_vjp, nondiff_argnums=(0,))
    @abstractmethod
    def _solve_vjp(self, problem: OptimizationProblem):
        pass
