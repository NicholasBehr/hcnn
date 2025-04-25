from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class OptimizationVariable(ABC):
    # INEQUALITY constraint relaxation to achieve feasibility
    # (problem was infeasible if ineq_relaxation > 0)
    max_ineq_relaxation: float
    ineq_relaxations: jax.Array

    # Addition
    def __add__(self, other):
        if isinstance(other, OptimizationVariable):
            return jax.tree_util.tree_map(lambda x, y: x + y, self, other)
        elif isinstance(other, (float, int)):
            return jax.tree_util.tree_map(lambda x: x + other, self)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    # Multiplication
    def __mul__(self, other):
        if isinstance(other, OptimizationVariable):
            return jax.tree_util.tree_map(lambda x, y: x * y, self, other)
        elif isinstance(other, (float, int)):
            return jax.tree_util.tree_map(lambda x: x * other, self)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    # Subtraction
    def __sub__(self, other):
        if isinstance(other, OptimizationVariable):
            return jax.tree_util.tree_map(lambda x, y: x - y, self, other)
        elif isinstance(other, (float, int)):
            return jax.tree_util.tree_map(lambda x: x - other, self)
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (float, int)):
            return jax.tree_util.tree_map(lambda x: other - x, self)
        else:
            return NotImplemented


@struct.dataclass
class OptimizationProblem(ABC):
    """
    Optimization Problem
    """

    @abstractmethod
    def warm_start(self) -> OptimizationVariable:
        """Use member variables to warm start the optimization variable."""
        pass

    ### QUADRATIC COST ###
    @abstractmethod
    def cost_matvec_quadratic(self, x: OptimizationVariable) -> OptimizationVariable:
        pass

    cost_linear: OptimizationVariable

    ### NONLINEAR CONSTRAINTS ###
    def constraints_equality(self, x: OptimizationVariable) -> jax.Array:
        """
        Defines equality constraints for an optimization problem.
        This function represents an equality constraint of the form:
        b <= h(x) <= b, where h(x) is a function of the optimization variable `x`,
        and `b` is the lower and upper bound simultaneously.

        Args:
            x (OptimizationVariable): The optimization variable.

        Returns:
            jax.Array: The result of the equality constraint function h(x).
        """
        return jnp.zeros((1,))

    def constraints_inequality(self, x: OptimizationVariable) -> jax.Array:
        """
        Defines inequality constraints for the optimization problem.
        This function represents an inequality constraint of the form:
        l <= g(x) <= u, where g(x) is a function of the optimization variable `x`,
        and `l` and `u` are the lower and upper bounds, respectively.

        Args:
            x (OptimizationVariable): The optimization variable.
        Returns:
            jax.Array: The result of the equality constraint function g(x).
        """
        return jnp.concatenate(
            (
                x.ineq_relaxations.ravel() - x.max_ineq_relaxation,
                x.ineq_relaxations.ravel() + x.max_ineq_relaxation,
                jnp.array([x.max_ineq_relaxation]),
            )
        )

    ### CONSTANT BOUNDS ###
    def bound_equality(self, x: OptimizationVariable) -> jax.Array:
        """
        Returns the equality bounds for the equality constraint.

        This method returns the lower and upper bounds for the equality constraints
        of the optimization problem. Naturally, they are always the same value.

        Returns:
            jax.Array: Array containing bound
            bounds for the equality constraints.
        """
        out = self.constraints_equality(x=x)
        return jnp.zeros_like(out)

    def bounds_inequality(self, x: OptimizationVariable) -> tuple[jax.Array, jax.Array]:
        """
        Returns the inequality bounds for the optimization variable.

        This method returns constant lower and upper bounds for the inequality
        constraints of the optimization problem.

        Returns:
            tuple[jax.Array, jax.Array]: A tuple containing the lower and upper
            bounds for the inequality constraints.
        """
        return (
            jnp.concatenate(
                (
                    -jnp.inf * jnp.ones((x.ineq_relaxations.size)),
                    jnp.zeros((x.ineq_relaxations.size + 1)),
                )
            ),
            jnp.concatenate(
                (
                    jnp.zeros((x.ineq_relaxations.size)),
                    jnp.inf * jnp.ones((x.ineq_relaxations.size + 1)),
                )
            ),
        )


@struct.dataclass
class QCNC(ABC):
    """
    Quadratic Cost with Nonlinear Constraints Problem Solver
    """

    @abstractmethod
    def solve(self, problem: OptimizationProblem) -> OptimizationVariable:
        pass
