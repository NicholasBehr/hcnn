from abc import ABC, abstractmethod
from functools import partial
from typing import Generic, Tuple, TypeVar

import jax
import jax.numpy as jnp
from flax import struct

PhiTypeDiff = TypeVar("PhiTypeDiff")
PhiTypeNoDiff = TypeVar("PhiTypeNoDiff")


@struct.dataclass
class OptimizationProblemBase(ABC, Generic[PhiTypeDiff, PhiTypeNoDiff]):
    """
    Optimization Problem Definition
    """

    ### INITIALIZATION ###
    # get initial guess for optimization variable x (vectorized 1d)
    @abstractmethod
    def get_x0(self, phi_d: PhiTypeDiff, phi_nd: PhiTypeNoDiff) -> jax.Array:
        pass

    ### QUADRATIC COST ###
    # f(x) = 1/2 x.T @ diag(q) @ x + c.T @ x
    cost_quad_diag: jax.Array  # q in cost (vectorized 1d)
    cost_linear: jax.Array  # c in cost (vectorized 1d)

    ### NONLINEAR PARAMETRIC CONSTRAINTS ###
    # l <= h(x, phi) <= u
    @abstractmethod
    def constraint_fn(
        self, x: jax.Array, phi_d: PhiTypeDiff, phi_nd: PhiTypeNoDiff
    ) -> jax.Array:
        pass

    lower_bound: jax.Array  # l in constraint (vectorized 1d)
    upper_bound: jax.Array  # u in constraint (vectorized 1d)


@struct.dataclass
class OptimizationVariables(struct.PyTreeNode):
    """
    Optimization Variables
    This class holds the current optimization variables and their residuals.
    """

    # primal
    x: jax.Array
    x_tilde: jax.Array
    z: jax.Array
    z_tilde: jax.Array
    primal_residuals: jax.Array
    # dual
    y: jax.Array
    dual_residuals: jax.Array
    # counter
    iter_counter: jax.Array

    @staticmethod
    def factory(
        problem: OptimizationProblemBase[PhiTypeDiff, PhiTypeNoDiff],
        phi_d: PhiTypeDiff,
        phi_nd: PhiTypeNoDiff,
    ) -> "OptimizationVariables":
        x = problem.get_x0(phi_d, phi_nd)
        z = problem.constraint_fn(x, phi_d, phi_nd)
        return OptimizationVariables(
            # primal
            x=x,
            x_tilde=x,
            z=z,
            z_tilde=z,
            primal_residuals=jnp.zeros_like(z),
            # dual
            y=jnp.zeros_like(z),
            dual_residuals=problem.cost_quad_diag * x + problem.cost_linear,
            # counter
            iter_counter=jnp.array(0, dtype=int),
        )


@struct.dataclass
class OptimizationInfo:
    """
    Optimization Information
    This class holds convergence information.
    """

    primal_residual: jax.Array
    dual_residual: jax.Array
    solver_iterations: jax.Array


@struct.dataclass
class OptimizationSolverBase(ABC, Generic[PhiTypeDiff, PhiTypeNoDiff]):
    """
    Quadratic Cost with Nonlinear Constraints Problem Solver
    """

    # relaxation parameter in range (0, 2)
    alpha: float = 1.6

    # dual step-size to align x and x_tilde
    sigma: float = 1e-2

    # dual step-size to align z and z_tilde
    rho_equality: float = 1e-1
    rho_inequality: float = 1e-1

    # maximum number of iterations
    max_iter: int = int(100)

    # convergence tolerance for primal and dual residuals
    tol: float = 1e-4

    # gradient clipping threshold
    gradient_clip: float = 100

    @partial(jax.jit, static_argnames="self")
    def solve(
        self,
        problem: OptimizationProblemBase,
        phi_d: PhiTypeDiff,
        phi_nd: PhiTypeNoDiff,
    ) -> Tuple[jax.Array, OptimizationInfo]:
        # argument "self" needs to be passed explicitly for custom vjp to work
        return self._solve_vjp(
            self, problem=problem, phi_d=phi_d, phi_nd=phi_nd
        )

    @abstractmethod
    def update_primal_dual(
        self,
        variables: OptimizationVariables,
        problem: OptimizationProblemBase[PhiTypeDiff, PhiTypeNoDiff],
        phi_d: PhiTypeDiff,
        phi_nd: PhiTypeNoDiff,
    ) -> OptimizationVariables:
        pass

    def update_residuals(
        self,
        variables: OptimizationVariables,
        problem: OptimizationProblemBase[PhiTypeDiff, PhiTypeNoDiff],
        phi_d: PhiTypeDiff,
        phi_nd: PhiTypeNoDiff,
    ) -> OptimizationVariables:
        ## form linearization around x
        h, h_vjp = jax.vjp(
            lambda x: problem.constraint_fn(x, phi_d, phi_nd), variables.x
        )

        primal_residuals = h - variables.z
        dual_residuals = (
            problem.cost_quad_diag * variables.x
            + problem.cost_linear
            + h_vjp(variables.y)[0]
        )

        return variables.replace(
            primal_residuals=primal_residuals,
            dual_residuals=dual_residuals,
        )

    def cond_fun(self, variables: OptimizationVariables) -> jax.Array:
        # return true if we should enter a new iteration

        # are we not converged?
        primal_check = jnp.max(jnp.abs(variables.primal_residuals)) >= self.tol
        dual_check = jnp.max(jnp.abs(variables.dual_residuals)) >= self.tol
        not_converged_check = jnp.logical_or(primal_check, dual_check)

        # AND do we have iterations left?
        iter_remaining_check = variables.iter_counter < self.max_iter

        return jnp.logical_and(not_converged_check, iter_remaining_check)

    def body_fun(
        self,
        variables: OptimizationVariables,
        problem: OptimizationProblemBase[PhiTypeDiff, PhiTypeNoDiff],
        phi_d: PhiTypeDiff,
        phi_nd: PhiTypeNoDiff,
    ) -> OptimizationVariables:
        # Update primal and dual variables
        variables = self.update_primal_dual(
            variables=variables,
            problem=problem,
            phi_d=phi_d,
            phi_nd=phi_nd,
        )

        # Update residuals
        variables = self.update_residuals(
            variables=variables,
            problem=problem,
            phi_d=phi_d,
            phi_nd=phi_nd,
        )

        # Increment iteration counter
        variables = variables.replace(iter_counter=variables.iter_counter + 1)

        return variables

    def _solve_variables(
        self,
        problem: OptimizationProblemBase[PhiTypeDiff, PhiTypeNoDiff],
        phi_d: PhiTypeDiff,
        phi_nd: PhiTypeNoDiff,
    ) -> Tuple[OptimizationVariables, OptimizationInfo]:
        variables = OptimizationVariables.factory(problem, phi_d, phi_nd)
        variables = jax.lax.while_loop(
            cond_fun=self.cond_fun,
            body_fun=partial(
                self.body_fun, problem=problem, phi_d=phi_d, phi_nd=phi_nd
            ),
            init_val=variables,
        )
        info = OptimizationInfo(
            primal_residual=jnp.max(jnp.abs(variables.primal_residuals)),
            dual_residual=jnp.max(jnp.abs(variables.dual_residuals)),
            solver_iterations=variables.iter_counter,
        )
        return variables, info

    def _solve(
        self,
        problem: OptimizationProblemBase[PhiTypeDiff, PhiTypeNoDiff],
        phi_d: PhiTypeDiff,
        phi_nd: PhiTypeNoDiff,
    ):
        variables, info = self._solve_variables(problem, phi_d, phi_nd)

        return (variables.x, info)

    def _solve_fwd(
        self,
        problem: OptimizationProblemBase[PhiTypeDiff, PhiTypeNoDiff],
        phi_d: PhiTypeDiff,
        phi_nd: PhiTypeNoDiff,
    ):
        variables, info = self._solve_variables(problem, phi_d, phi_nd)
        return (variables.x, info), (
            variables.x,
            variables.z,
            variables.y,
            problem,
            phi_d,
            phi_nd,
        )

    def _solve_bwd(
        self,
        res: Tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            OptimizationProblemBase[PhiTypeDiff, PhiTypeNoDiff],
            PhiTypeDiff,
            PhiTypeNoDiff,
        ],
        g: jax.Array,
    ):
        ## Unpack and rename input
        x_star, z_star, y_star, problem, phi_d, phi_nd = res
        y_plus = jnp.maximum(y_star, 0)
        y_minus = jnp.minimum(y_star, 0)
        v1 = g[0]  # gradient w.r.t. x_star

        ## Implicit VJP (step 1) - Add numerical stability
        # Add tol to prevent division by zero
        # (since this is the tolerance of the divisor anyways)
        frac_upper = jax.lax.div(1.0, problem.upper_bound - z_star + self.tol)
        frac_lower = jax.lax.div(1.0, z_star - problem.lower_bound + self.tol)
        plus_u_minus_l = y_plus * frac_upper - y_minus * frac_lower

        ## Precompute JVP, VJP and HVP for the constraint function
        # h(x) = problem.constraint_fn_x(x)
        # h_jvp computes v -> dh/dx @ v
        _, h_jvp = jax.linearize(
            lambda x: problem.constraint_fn(x, phi_d, phi_nd), x_star
        )
        # h_vjp computes w -> w^T @ dh/dx
        _, h_vjp = jax.vjp(
            lambda x: problem.constraint_fn(x, phi_d, phi_nd), x_star
        )
        # h_hvp computes y.T @ d^2h/dx^2 @ v
        _, h_hvp = jax.linearize(
            jax.grad(
                lambda x: jnp.dot(
                    y_star, problem.constraint_fn(x, phi_d, phi_nd)
                )
            ),
            x_star,
        )

        def schur_lhs(u1: jax.Array) -> jax.Array:
            return (
                problem.cost_quad_diag * u1
                + h_hvp(u1)
                + h_vjp(plus_u_minus_l * h_jvp(u1))[0]
            )

        # Use more robust solver settings
        u1, _ = jax.scipy.sparse.linalg.bicgstab(A=schur_lhs, b=v1)

        # replace nan with zeros
        u1 = jnp.nan_to_num(u1, nan=0.0, posinf=1e4, neginf=-1e4)

        h_jvp_u1 = h_jvp(u1)
        u2 = plus_u_minus_l * h_jvp_u1
        u3 = frac_upper * h_jvp_u1
        u4 = frac_lower * h_jvp_u1

        ## Explicit VJP (step 2)
        g_problem = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x), problem
        )  # copy
        g_problem = g_problem.replace(
            cost_quad_diag=-x_star * u1,
            cost_linear=-u1,  # type: ignore -> is always jax.array
            upper_bound=+y_plus * u3,
            lower_bound=-y_minus * u4,
        )

        g_phi_d_1 = jax.grad(
            lambda phi_d_arg: jnp.dot(
                jax.jvp(
                    lambda x_inner: problem.constraint_fn(
                        x_inner, phi_d_arg, phi_nd
                    ),
                    (x_star,),
                    (-u1,),  # type: ignore -> is always jax.array
                )[1],
                y_star,
            ),
        )(phi_d)
        g_phi_d_2 = jax.vjp(
            lambda phi_d_arg: problem.constraint_fn(x_star, phi_d_arg, phi_nd),
            phi_d,
        )[1](-u2)[0]
        g_phi_d = jax.tree_util.tree_map(jnp.add, g_phi_d_1, g_phi_d_2)

        # Add gradient clipping
        g_problem = jax.tree_util.tree_map(
            lambda g: jnp.clip(g, -self.gradient_clip, self.gradient_clip),
            g_problem,
        )
        g_phi_d = jax.tree_util.tree_map(
            lambda g: jnp.clip(g, -self.gradient_clip, self.gradient_clip),
            g_phi_d,
        )

        return (g_problem, g_phi_d, None)

    _solve_vjp = jax.custom_vjp(_solve, nondiff_argnums=(0,))
    _solve_vjp.defvjp(_solve_fwd, _solve_bwd)
