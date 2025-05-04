from abc import ABC, abstractmethod

import jax
from flax import struct


@struct.dataclass
class OptimizationProblem(ABC):
    """
    Optimization Problem Definition
    """

    cost_linear: jax.Array  # c in cost definition below (vectorized 1d)
    lower_bound: jax.Array  # l in constraint definition below (vectorized 1d)
    upper_bound: jax.Array  # u in constraint definition below (vectorized 1d)

    ### OPTIMIZATION VARIABLE ###
    # property allows to abstact away the actual storage format of x (usually is struct)
    @property
    @abstractmethod
    def x0(self) -> jax.Array:  # initial optimization variable (vectorized 1d)
        pass

    ### QUADRATIC COST ###
    # f(x) = 1/2 x.T @ Q @ x + c.T @ x

    # Q in cost definition above
    @abstractmethod
    def cost_qvp(self, x: jax.Array) -> jax.Array:
        """
        Memory efficient representation of Q
        Q must be PSD!

        Args:
            x (jax.Array): The optimization variable.
        Returns:
            jax.Array: The result of the computation Q @ x
        """
        pass

    def cost_fun(self, x: jax.Array) -> jax.Array:
        return 1 / 2 * x.T @ self.cost_qvp(x) + self.cost_linear.T @ x

    ### NONLINEAR CONSTRAINTS ###
    @abstractmethod
    def constraints(self, x: jax.Array) -> jax.Array:
        """
        Defines constraints for the optimization problem.
        This function represents an inequality constraint of the form:
        l <= g(x) <= u, where g(x) is a function of the optimization variable `x`,
        and `l` and `u` are the `lower_bound` and `upper_bound`, respectively.

        Args:
            x (jax.Array): The optimization variable.
        Returns:
            jax.Array: The result of the constraint function g(x).
        """
        pass


@struct.dataclass
class OptimizationSolver(ABC):
    """
    Quadratic Cost with Nonlinear Constraints Problem Solver
    """

    @abstractmethod
    def solve(self, problem: OptimizationProblem) -> jax.Array:
        pass
