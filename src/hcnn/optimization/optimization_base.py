from abc import ABC, abstractmethod
from functools import partial
from typing import Generic, TypeVar

import jax
from flax import struct

PhiType = TypeVar("PhiType")


@struct.dataclass
class OptimizationProblemBase(ABC, Generic[PhiType]):
    """
    Optimization Problem Definition
    """

    ### INITIALIZATION ###
    # get initial guess for optimization variable x (vectorized 1d)
    @staticmethod
    @abstractmethod
    def get_x0(phi: PhiType) -> jax.Array:
        pass

    ### QUADRATIC COST ###
    # f(x) = 1/2 x.T @ diag(q) @ x + c.T @ x
    cost_quad_diag: jax.Array  # q in cost (vectorized 1d)
    cost_linear: jax.Array  # c in cost (vectorized 1d)

    ### NONLINEAR PARAMETRIC CONSTRAINTS ###
    # l <= h(x, phi) <= u
    @staticmethod
    @abstractmethod
    def constraint_fn(x: jax.Array, phi: PhiType) -> jax.Array:
        pass

    lower_bound: jax.Array  # l in constraint (vectorized 1d)
    upper_bound: jax.Array  # u in constraint (vectorized 1d)


@struct.dataclass
class OptimizationSolverBase(ABC, Generic[PhiType]):
    """
    Quadratic Cost with Nonlinear Constraints Problem Solver
    """

    @partial(jax.jit, static_argnames='self')
    def solve(self, problem: OptimizationProblemBase, phi: PhiType) -> jax.Array:
        """
        Solve the given optimization problem and return the optimal solution.

        This method takes an `OptimizationProblem` instance containing:
        - a quadratic cost function
        - nonlinear parametric constraints
        - an initial guess for the optimal decision variable `x*`

        TODO explain context

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
        return self._solve_vjp(self, problem=problem, phi=phi)

    @partial(jax.custom_vjp, nondiff_argnums=(0,))
    @abstractmethod
    def _solve_vjp(self, problem: OptimizationProblemBase, phi: PhiType) -> jax.Array:
        pass
