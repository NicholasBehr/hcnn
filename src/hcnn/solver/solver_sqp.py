from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Generic, TypeVar

import jax
import jax.numpy as jnp
from flax import struct

from .solver_osqp import OSQPParameters, OSQPProblem, osqp_solve

PhiType = TypeVar("PhiType")


@struct.dataclass
class SQPInfo:
    primal_residual: jax.Array
    dual_residual: jax.Array
    iterations_lin: jax.Array
    iterations_nonlin: jax.Array
    rho_base_mult: jax.Array
    converged_flag: jax.Array
    avg_step_size: jax.Array  # Average line search step size


@struct.dataclass
class SQPProblemBase(Generic[PhiType], struct.PyTreeNode, ABC):
    ### INITIALIZATION ###
    # get initial guess for optimization variable x (vectorized 1d)
    @abstractmethod
    def get_x0(self, phi: PhiType) -> jax.Array:
        pass

    @property
    def x0(self):
        return self.get_x0(self.phi)

    ### QUADRATIC COST ###
    # f(x) = 1/2 x.T @ diag(q) @ x + c.T @ x
    cost_quad_diag: jax.Array  # q in cost (vectorized 1d)
    cost_linear: jax.Array  # c in cost (vectorized 1d)

    ### NONLINEAR PARAMETRIC CONSTRAINTS ###
    # l <= h(x, phi) <= u
    @abstractmethod
    def constraint_fn_phi(self, x: jax.Array, phi: PhiType) -> jax.Array:
        pass

    def constraint_fn(self, x: jax.Array) -> jax.Array:
        return self.constraint_fn_phi(x, self.phi)

    phi: PhiType
    lower_bound: jax.Array  # l in constraint (vectorized 1d)
    upper_bound: jax.Array  # u in constraint (vectorized 1d)


@struct.dataclass
class SQPParameters:
    alpha: float = struct.field(default_factory=lambda: 1.6)
    sigma: float = struct.field(default_factory=lambda: 1e-6)
    rho: float = struct.field(default_factory=lambda: 0.1)
    rho_equality_mult: float = struct.field(default_factory=lambda: 1e3)
    tol_prim: float = struct.field(default_factory=lambda: 1e-4)
    tol_dual: float = struct.field(default_factory=lambda: 1e-4)
    tol_osqp: float = struct.field(default_factory=lambda: 1e-5)
    epsilon: float = struct.field(default_factory=lambda: 1e-12)
    max_iter_lin: int = struct.field(default_factory=lambda: 2e3)
    max_iter_nonlin: int = struct.field(default_factory=lambda: 10)
    # Line search parameters
    ls_c1: float = struct.field(default_factory=lambda: 1e-4)
    ls_rho: float = struct.field(default_factory=lambda: 0.75)
    ls_max_iter: int = struct.field(default_factory=lambda: 20)
    ls_penalty_weight: float = struct.field(default_factory=lambda: 100)


@struct.dataclass
class SQPVariables(struct.PyTreeNode):
    x: jax.Array
    z: jax.Array
    y: jax.Array
    max_primal_residual: jax.Array
    max_dual_residual: jax.Array
    q_plus_s: jax.Array
    nonlin_iter: jax.Array
    sumlin_iter: jax.Array
    sum_rho_base_mult: jax.Array
    sum_step_size: jax.Array  # Sum of line search step sizes
    is_infeasible: jax.Array

    @staticmethod
    def factory(problem: SQPProblemBase, parameters: SQPParameters):
        x = problem.x0
        h = problem.constraint_fn(x)
        z = jnp.clip(h, problem.lower_bound, problem.upper_bound)
        y = jnp.zeros_like(z)

        q_plus_s = jnp.maximum(problem.cost_quad_diag, parameters.epsilon)

        # h(x) - z
        primal_residuals = h - z
        # Qx + c + A.T y
        dual_residuals = q_plus_s * x + problem.cost_linear

        return SQPVariables(
            x=x,
            z=z,
            y=y,
            max_primal_residual=jnp.max(jnp.abs(primal_residuals)),
            max_dual_residual=jnp.max(jnp.abs(dual_residuals)),
            q_plus_s=q_plus_s,
            nonlin_iter=jnp.array(0),
            sumlin_iter=jnp.array(0),
            sum_rho_base_mult=jnp.array(0),
            sum_step_size=jnp.array(0.0),
            is_infeasible=jnp.array(False),
        )


def compute_hessian_diag(
    constraint_fun: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    y: jax.Array,
):
    """
    Computes diag(Hessian_x(y.T @ h(x)))
    """

    # 1. Define the scalar-valued function g(x) and its gradient function
    def g(x_arg):
        return jnp.dot(y, constraint_fun(x_arg))

    grad_g = jax.grad(g)

    # 2. Define a function that computes one diagonal element (H_jj)
    #    H_jj is the j-th element of the HVP with the j-th basis vector e_j.
    @partial(jax.vmap, in_axes=(0))
    def get_diag_element(e_j):
        # jax.jvp computes the product of the Jacobian of grad_g (the Hessian)
        # with the vector e_j. It returns the primal output and the tangent
        # output (the HVP).
        hvp = jax.jvp(grad_g, (x,), (e_j,))[1]
        # We only need the j-th component of the HVP result.
        return jnp.dot(hvp, e_j)

    # 3. Create the standard basis vectors (identity matrix) and apply the
    # vmapped function
    n = x.shape[0]
    eye = jnp.eye(n, dtype=x.dtype)
    return get_diag_element(eye)


def compute_merit_function(
    x: jax.Array,
    problem: SQPProblemBase,
    penalty_weight: float,
) -> jax.Array:
    """
    Computes L1 merit function: f(x) + penalty_weight * ||h(x) - proj(h(x))||_1
    where proj(h(x)) projects h(x) onto the feasible set [l, u].
    """
    # Quadratic cost: 1/2 x.T @ diag(q) @ x + c.T @ x
    cost = 0.5 * jnp.sum(problem.cost_quad_diag * x * x) + jnp.dot(
        problem.cost_linear, x
    )

    # Constraint violations
    h = problem.constraint_fn(x)
    projected_h = jnp.clip(h, problem.lower_bound, problem.upper_bound)
    violation = jnp.sum(jnp.abs(h - projected_h))

    return cost + penalty_weight * violation


def compute_merit_gradient_dot_dx(
    x: jax.Array,
    dx: jax.Array,
    problem: SQPProblemBase,
    penalty_weight: float,
) -> jax.Array:
    """
    Computes the directional derivative of the merit function:
    grad_merit(x).T @ dx
    """
    # Gradient of quadratic cost: q * x + c
    cost_grad = problem.cost_quad_diag * x + problem.cost_linear
    cost_grad_dot_dx = jnp.dot(cost_grad, dx)

    # Constraint violations
    h = problem.constraint_fn(x)
    projected_h = jnp.clip(h, problem.lower_bound, problem.upper_bound)
    violation = jnp.sum(jnp.abs(h - projected_h))

    return cost_grad_dot_dx - penalty_weight * violation


def line_search_step(
    x_current: jax.Array,
    dx: jax.Array,
    problem: SQPProblemBase,
    parameters: SQPParameters,
    merit_current: jax.Array,
    merit_grad_dot_dx: jax.Array,
) -> jax.Array:
    """
    Performs backtracking line search to find step size beta.
    Returns the step size that satisfies Armijo condition.
    """
    min_beta = 1e-12  # Minimum step size to avoid numerical issues

    def line_search_cond(state):
        beta, iteration, armijo_satisfied = state
        # Continue if: not satisfied AND within limits AND beta not too small
        return jnp.logical_and(
            jnp.logical_not(armijo_satisfied),
            jnp.logical_and(
                iteration < parameters.ls_max_iter,
                beta > min_beta,
            ),
        )

    def line_search_body(state):
        beta, iteration, _ = state

        # Trial point
        x_trial = x_current + beta * dx
        merit_trial = compute_merit_function(
            x_trial, problem, parameters.ls_penalty_weight
        )

        # Armijo condition
        armijo_rhs = merit_current + parameters.ls_c1 * beta * merit_grad_dot_dx
        armijo_satisfied = merit_trial <= armijo_rhs

        # Update step size if Armijo condition not satisfied
        new_beta = jnp.where(armijo_satisfied, beta, beta * parameters.ls_rho)

        return new_beta, iteration + 1, armijo_satisfied

    # Initialize with full step
    init_state = (jnp.array(1.0), jnp.array(0), jnp.array(False))
    final_beta, _, _ = jax.lax.while_loop(
        line_search_cond, line_search_body, init_state
    )

    return final_beta


def sqp_step(
    variables: SQPVariables,
    problem: SQPProblemBase,
    parameters: SQPParameters,
) -> SQPVariables:
    ## q ##
    hessian_diag = compute_hessian_diag(
        constraint_fun=problem.constraint_fn, x=variables.x, y=variables.y
    )
    new_q_plus_s = jnp.maximum(
        problem.cost_quad_diag + hessian_diag, parameters.epsilon
    )

    ## c ##
    new_c = problem.cost_quad_diag * variables.x + problem.cost_linear

    ## bounds ##
    offset = problem.constraint_fn(variables.x)
    new_l = problem.lower_bound - offset
    new_u = problem.upper_bound - offset

    ## Solve for step using OSQP ##
    h_jvp = jax.linearize(problem.constraint_fn, variables.x)[1]
    h_vjp = jax.vjp(problem.constraint_fn, variables.x)[1]
    osqp_problem = OSQPProblem(
        x0=jnp.zeros_like(variables.x),
        cost_quad_diag=new_q_plus_s,
        cost_linear=new_c,
        lower_bound=new_l,
        upper_bound=new_u,
        matrix_vector_product=h_jvp,
        vector_matrix_product=lambda z: h_vjp(z)[0],
    )
    osqp_parameters = OSQPParameters(
        alpha=parameters.alpha,
        sigma=parameters.sigma,
        rho=parameters.rho,
        rho_equality_mult=parameters.rho_equality_mult,
        tol_prim=parameters.tol_osqp,
        tol_dual=parameters.tol_osqp,
        max_iter=parameters.max_iter_lin,
    )
    osqp_variables = osqp_solve(osqp_problem, osqp_parameters)

    ## Line search for step size ##
    dx = osqp_variables.x
    dy = osqp_variables.y - variables.y

    # Current merit function value
    merit_current = compute_merit_function(
        variables.x, problem, parameters.ls_penalty_weight
    )

    # Directional derivative of merit function
    merit_grad_dot_dx = compute_merit_gradient_dot_dx(
        variables.x, dx, problem, parameters.ls_penalty_weight
    )

    # Find optimal step size
    beta = line_search_step(
        variables.x,
        dx,
        problem,
        parameters,
        merit_current,
        merit_grad_dot_dx,
    )

    # jax.debug.print(
    #     "SQP iter {}/{} | beta: {:.3e} | lin iters: {}",
    #     variables.nonlin_iter + 1,
    #     parameters.max_iter_nonlin,
    #     beta,
    #     osqp_variables.iteration,
    # )

    # Update variables with line search step size
    new_x = variables.x + beta * dx
    new_z = jnp.clip(
        problem.constraint_fn(new_x), problem.lower_bound, problem.upper_bound
    )
    new_y = variables.y + beta * dy

    ## residuals ##
    new_h, new_h_vjp = jax.vjp(problem.constraint_fn, new_x)
    primal_residuals = new_h - new_z
    dual_residuals = (
        problem.cost_quad_diag * new_x
        + problem.cost_linear
        + new_h_vjp(new_y)[0]
    )

    ## iteration ##
    new_nonlin_iter = variables.nonlin_iter + 1
    new_sumlin_iter = variables.sumlin_iter + osqp_variables.iteration

    ## log multiplier ##
    new_sum_rho_base_mult = (
        variables.sum_rho_base_mult + osqp_variables.rho_base_mult
    )

    ## log step size ##
    new_sum_step_size = variables.sum_step_size + beta

    return SQPVariables(
        x=new_x,
        z=new_z,
        y=new_y,
        max_primal_residual=jnp.max(jnp.abs(primal_residuals)),
        max_dual_residual=jnp.max(jnp.abs(dual_residuals)),
        q_plus_s=new_q_plus_s,
        nonlin_iter=new_nonlin_iter,
        sumlin_iter=new_sumlin_iter,
        sum_rho_base_mult=new_sum_rho_base_mult,
        sum_step_size=new_sum_step_size,
        is_infeasible=variables.is_infeasible,
    )


def sqp_not_converged(
    variables: SQPVariables,
    parameters: SQPParameters,
) -> jax.Array:
    # Return early if problem is detected as infeasible
    infeasible = variables.is_infeasible

    # check if primal or dual residuals are above the tolerance
    primal = variables.max_primal_residual >= parameters.tol_prim
    dual = variables.max_dual_residual >= parameters.tol_dual
    residual = jnp.logical_or(primal, dual)

    # check if maximum number of iterations has not been reached
    iteration = variables.nonlin_iter < parameters.max_iter_nonlin

    # declare not converged if residuals are high AND iterations remain
    # BUT return converged (False) if infeasible is detected
    not_converged_by_residual = jnp.logical_and(residual, iteration)

    # Return True (not converged) only if not infeasible and residuals/iter
    return jnp.logical_and(
        jnp.logical_not(infeasible), not_converged_by_residual
    )


def sqp_solve_variables(
    problem: SQPProblemBase,
    parameters: SQPParameters,
) -> tuple[SQPVariables, SQPInfo]:
    variables = jax.lax.while_loop(
        cond_fun=partial(
            sqp_not_converged,
            parameters=parameters,
        ),
        body_fun=partial(
            sqp_step,
            problem=problem,
            parameters=parameters,
        ),
        init_val=SQPVariables.factory(
            problem=problem,
            parameters=parameters,
        ),
    )
    sqp_info = SQPInfo(
        primal_residual=variables.max_primal_residual,
        dual_residual=variables.max_dual_residual,
        iterations_lin=variables.sumlin_iter
        / jnp.maximum(variables.nonlin_iter, 1),
        iterations_nonlin=variables.nonlin_iter,
        rho_base_mult=variables.sum_rho_base_mult
        / jnp.maximum(variables.nonlin_iter, 1),
        converged_flag=jnp.logical_and(
            variables.max_primal_residual < parameters.tol_prim,
            variables.max_dual_residual < parameters.tol_dual,
        ),
        avg_step_size=variables.sum_step_size
        / jnp.maximum(variables.nonlin_iter, 1),
    )

    return variables, sqp_info


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def sqp_solve(
    problem: SQPProblemBase,
    parameters: SQPParameters,
) -> tuple[jax.Array, SQPInfo]:
    problem_reg = problem.replace(
        cost_quad_diag=jnp.maximum(problem.cost_quad_diag, parameters.epsilon)
    )
    variables, sqp_info = sqp_solve_variables(
        problem=problem_reg,
        parameters=parameters,
    )
    return variables.x, sqp_info


def sqp_solve_fwd(
    problem: SQPProblemBase,
    parameters: SQPParameters,
) -> tuple[
    tuple[jax.Array, SQPInfo],
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, SQPProblemBase],
]:
    problem_reg = problem.replace(
        cost_quad_diag=jnp.maximum(problem.cost_quad_diag, parameters.epsilon)
    )
    variables, sqp_info = sqp_solve_variables(
        problem=problem_reg,
        parameters=parameters,
    )

    return (
        (variables.x, sqp_info),
        (
            variables.x,  # primal solution
            variables.y,  # dual variables
            variables.q_plus_s,  # diagonal Hessian approx
            sqp_info.converged_flag,  # converged flag
            problem_reg,  # problem data (differentiable arg structure)
        ),
    )


def sqp_solve_bwd(
    parameters: SQPParameters,
    res: tuple[jax.Array, jax.Array, jax.Array, jax.Array, SQPProblemBase],
    grad: tuple[jax.Array, SQPInfo],
):
    x_star, y_star, q_plus_s, converged_flag, problem = res
    v1, _ = grad

    def compute_zero_gradients():
        return (jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), problem),)

    def compute_vjp_gradients():
        ## Precompute JVP, VJP and HVP for the constraint function
        # h_jvp computes v -> dh/dx @ v
        h, h_jvp = jax.linearize(problem.constraint_fn, x_star)
        # h_vjp computes w -> w^T @ dh/dx
        _, h_vjp = jax.vjp(problem.constraint_fn, x_star)

        ## Implicit VJP (step 1)
        y_plus = jnp.maximum(y_star, 0.0)
        y_minus = jnp.maximum(-y_star, 0.0)

        reg = 10 * parameters.tol_prim
        frac_upper = y_plus / (reg + problem.upper_bound - h)
        frac_lower = y_minus / (reg + h - problem.lower_bound)
        frac_sum = frac_upper + frac_lower

        def schur_lhs(u1: jax.Array) -> jax.Array:
            return q_plus_s * u1 + h_vjp(frac_sum * h_jvp(u1))[0]

        u1, _ = jax.scipy.sparse.linalg.bicgstab(A=schur_lhs, b=-v1)
        u1 = jnp.where(jnp.isnan(u1), 0.0, u1)

        jvp_u1 = h_jvp(u1)
        u2 = jvp_u1 / (reg + problem.upper_bound - h)
        u3 = jvp_u1 / (problem.lower_bound - reg - h)

        ## Explicit VJP (step 2)
        grad_problem = jax.tree.map(jnp.zeros_like, problem)  # empty copy

        g_q = u1 * x_star
        g_c = u1
        g_u = u2 * (-y_plus)
        g_l = u3 * (+y_minus)

        ## phi grad
        g_phi_d_1 = jax.grad(
            lambda phi_arg: jnp.dot(
                jax.jvp(
                    lambda x_inner: problem.constraint_fn_phi(x_inner, phi_arg),
                    (x_star,),
                    (u1,),
                )[1],
                y_star,
            ),
        )(problem.phi)

        g_phi_d_2 = jax.vjp(
            lambda phi_arg: problem.constraint_fn_phi(x_star, phi_arg),
            problem.phi,
        )[1](u2 * y_plus - u3 * y_minus)[0]
        g_phi_d = jax.tree_util.tree_map(jnp.add, g_phi_d_1, g_phi_d_2)

        # populate struct
        grad_problem = grad_problem.replace(
            cost_quad_diag=g_q,
            cost_linear=g_c,
            phi=g_phi_d,
            lower_bound=g_l,
            upper_bound=g_u,
        )

        return (grad_problem,)

    # Return zero gradient if fixpoint wasnt found, otherwise compute VJP
    return jax.lax.cond(
        converged_flag,
        compute_vjp_gradients,
        compute_zero_gradients,
    )


sqp_solve.defvjp(sqp_solve_fwd, sqp_solve_bwd)
