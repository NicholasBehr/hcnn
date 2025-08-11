from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Generic, Tuple, TypeVar

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
    iter_nonlin: jax.Array
    iter_lin: jax.Array
    iter_lin_total: jax.Array

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
            iter_nonlin=jnp.array(0, dtype=int),
            iter_lin=jnp.array(0, dtype=int),
            iter_lin_total=jnp.array(0, dtype=int),
        )


@struct.dataclass
class OptimizationInfo:
    """
    Optimization Information
    This class holds convergence information.
    """

    primal_residual: jax.Array
    dual_residual: jax.Array
    iterations_lin: jax.Array
    iterations_nonlin: jax.Array
    converged_flag: jax.Array


@struct.dataclass
class ConstraintLinearization:
    x_lin: jax.Array
    # h(x_lin)
    h: jax.Array
    # dh/dx_lin h(x_lin)
    h_jvp: Callable[[jax.Array], jax.Array]
    h_vjp: Callable[[jax.Array], jax.Array]

    @classmethod
    def factory(
        cls,
        constraint_fn: Callable[[jax.Array], jax.Array],
        x_lin: jax.Array,
    ) -> "ConstraintLinearization":
        h, h_jvp = jax.linearize(constraint_fn, x_lin)
        _, h_vjp = jax.vjp(constraint_fn, x_lin)
        return cls(x_lin=x_lin, h=h, h_jvp=h_jvp, h_vjp=h_vjp)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Evaluate the linearized constraint function at x.
        """
        return self.h + self.h_jvp(x - self.x_lin)


@struct.dataclass
class OptimizationSolverBase(ABC, Generic[PhiTypeDiff, PhiTypeNoDiff]):
    """
    Quadratic Cost with Nonlinear Constraints Problem Solver
    """

    # relaxation parameter in range (0, 2)
    alpha: float = 1.6

    # dual step-size to align x and x_tilde
    sigma: float = 0.1

    # dual step-size to align z and z_tilde
    rho_equality: float = 100
    rho_inequality: float = 0.1

    # maximum number of iterations
    max_iter_nonlin: int = int(500)
    max_iter_lin: int = int(1e3)

    # convergence tolerance for primal and dual residuals
    tol: float = 1e-3

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
        constraint_lin: ConstraintLinearization,
    ) -> OptimizationVariables:
        pass

    def update_residuals(
        self,
        variables: OptimizationVariables,
        problem: OptimizationProblemBase[PhiTypeDiff, PhiTypeNoDiff],
        constraint_lin: ConstraintLinearization,
    ) -> OptimizationVariables:
        """
        Updates the primal and dual residuals of the optimization variables.
        With the primal residuals defined as:
            h(x) - z
        and the dual residuals defined as:
            Q x + c + (d/dx h(x))^T y
        """
        primal_residuals = constraint_lin(variables.x) - variables.z
        dual_residuals = (
            problem.cost_quad_diag * variables.x
            + problem.cost_linear
            + constraint_lin.h_vjp(variables.y)[0]
        )

        return variables.replace(
            primal_residuals=primal_residuals,
            dual_residuals=dual_residuals,
        )

    def convergence_check(self, variables: OptimizationVariables) -> jax.Array:
        """
        ASSUMES recent call to update_residuals() has been made!
        Returns:
            jax.Array: True if any convergence criteria are not met
            (i.e., optimization should continue).
        """
        # Check if primal and dual residuals are below the tolerance
        primal_check = jnp.max(jnp.abs(variables.primal_residuals)) >= self.tol
        dual_check = jnp.max(jnp.abs(variables.dual_residuals)) >= self.tol
        # Check if x is close to x_tilde
        x_tilde_check = (
            jnp.max(jnp.abs(variables.x - variables.x_tilde)) >= self.tol
        )

        return jnp.logical_or(
            jnp.logical_or(primal_check, dual_check), x_tilde_check
        )

    def cond_linear(self, variables: OptimizationVariables):
        """
        ASSUMES recent call to update_residuals() has been made!
        Returns:
            jax.Array: True if linear convergence criteria are not met
            (i.e., optimization should continue).
        """
        # are we not converged?
        not_converged_check = self.convergence_check(variables)
        # AND do we have iterations left?
        iter_remaining_check = variables.iter_lin < self.max_iter_lin

        return jnp.logical_and(not_converged_check, iter_remaining_check)

    def cond_nonlinear(self, variables: OptimizationVariables):
        """
        ASSUMES recent call to update_residuals() has been made!
        Returns:
            jax.Array: True if nonlinear convergence criteria are not met
            (i.e., optimization should continue).
        """
        # are we not converged?
        not_converged_check = self.convergence_check(variables)
        # AND do we have iterations left?
        iter_remaining_check = variables.iter_nonlin < self.max_iter_nonlin

        return jnp.logical_and(not_converged_check, iter_remaining_check)

    def body_linear(
        self,
        variables: OptimizationVariables,
        problem: OptimizationProblemBase[PhiTypeDiff, PhiTypeNoDiff],
        constraint_lin: ConstraintLinearization,
    ):
        group_steps = 50
        # Update primal and dual variables
        variables = jax.lax.fori_loop(
            lower=0,
            upper=group_steps,
            body_fun=lambda _, vars: self.update_primal_dual(
                variables=vars,
                problem=problem,
                constraint_lin=constraint_lin,
            ),
            init_val=variables,
            unroll=True,
        )

        # Update residuals
        variables = self.update_residuals(
            variables=variables,
            problem=problem,
            constraint_lin=constraint_lin,
        )

        # Increment iteration counter
        variables = variables.replace(
            iter_lin=variables.iter_lin + group_steps,
        )

        return variables

    def body_nonlinear(
        self,
        variables: OptimizationVariables,
        problem: OptimizationProblemBase[PhiTypeDiff, PhiTypeNoDiff],
        constraint_fn: Callable[[jax.Array], jax.Array],
    ) -> OptimizationVariables:
        # Update constraint linearization
        constraint_lin = ConstraintLinearization.factory(
            constraint_fn=constraint_fn,
            x_lin=variables.x_tilde,
        )

        # Solve linear QP subproblem
        variables = jax.lax.while_loop(
            cond_fun=self.cond_linear,
            body_fun=partial(
                self.body_linear,
                problem=problem,
                constraint_lin=constraint_lin,
            ),
            init_val=variables,
        )

        # Reset iter_lin
        variables = variables.replace(
            iter_lin=jnp.array(0, dtype=int),
            iter_lin_total=variables.iter_lin_total + variables.iter_lin,
        )

        # Update constraint linearization
        constraint_lin = ConstraintLinearization.factory(
            constraint_fn=constraint_fn,
            x_lin=variables.x,
        )

        # Update residuals
        variables = self.update_residuals(
            variables=variables,
            problem=problem,
            constraint_lin=constraint_lin,
        )

        # Increment iteration counter
        variables = variables.replace(
            iter_nonlin=variables.iter_nonlin + 1,
        )

        return variables

    def _solve_variables(
        self,
        problem: OptimizationProblemBase[PhiTypeDiff, PhiTypeNoDiff],
        phi_d: PhiTypeDiff,
        phi_nd: PhiTypeNoDiff,
    ) -> Tuple[OptimizationVariables, OptimizationInfo]:
        # Initialize optimization variables
        variables = OptimizationVariables.factory(problem, phi_d, phi_nd)

        # Capture phi_d and phi_nd in the constraint function closure
        constraint_fn = partial(
            problem.constraint_fn, phi_d=phi_d, phi_nd=phi_nd
        )

        # Calculate nonlinear fixedpoint
        variables = jax.lax.while_loop(
            cond_fun=self.cond_nonlinear,
            body_fun=partial(
                self.body_nonlinear,
                problem=problem,
                constraint_fn=constraint_fn,
            ),
            init_val=variables,
        )

        # Populate optimization info
        info = OptimizationInfo(
            primal_residual=jnp.max(jnp.abs(variables.primal_residuals)),
            dual_residual=jnp.max(jnp.abs(variables.dual_residuals)),
            iterations_lin=jnp.where(
                variables.iter_nonlin == 0,
                jnp.array(0.0),
                variables.iter_lin_total / variables.iter_nonlin,
            ),
            iterations_nonlin=variables.iter_nonlin,
            converged_flag=jnp.logical_not(self.convergence_check(variables)),
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
            info.converged_flag,
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
            bool,
        ],
        g: jax.Array,
    ):
        ## Unpack and rename input
        x_star, z_star, y_star, problem, phi_d, phi_nd, converged_flag = res
        y_plus = jnp.maximum(y_star, 0)
        y_minus = jnp.minimum(y_star, 0)
        v1 = g[0]  # gradient w.r.t. x_star

        def compute_zero_gradients():
            return (
                jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), problem),
                jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), phi_d),
                None,
            )

        def compute_vjp_gradients():
            ## Implicit VJP (step 1)
            frac_upper = jax.lax.div(
                1.0, problem.upper_bound - z_star + self.tol
            )
            frac_lower = jax.lax.div(
                1.0, z_star - problem.lower_bound + self.tol
            )
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
                    # + h_hvp(u1)  # TODO if added select GMRES solver
                    + h_vjp(plus_u_minus_l * h_jvp(u1))[0]
                    + 1e-12 * u1
                )

            # Solve the Schur complement system
            u1, _ = jax.scipy.sparse.linalg.cg(A=schur_lhs, b=v1)

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
                cost_linear=-u1,
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
                        (-u1,),
                    )[1],
                    y_star,
                ),
            )(phi_d)
            g_phi_d_2 = jax.vjp(
                lambda phi_d_arg: problem.constraint_fn(
                    x_star, phi_d_arg, phi_nd
                ),
                phi_d,
            )[1](-u2)[0]
            g_phi_d = jax.tree_util.tree_map(jnp.add, g_phi_d_1, g_phi_d_2)

            return (g_problem, g_phi_d, None)

        ## Return zero gradient if fixpoint wasnt found, otherwise compute VJP
        return jax.lax.cond(
            converged_flag,
            compute_vjp_gradients,
            compute_zero_gradients,
        )

    _solve_vjp = jax.custom_vjp(_solve, nondiff_argnums=(0,))
    _solve_vjp.defvjp(_solve_fwd, _solve_bwd)
