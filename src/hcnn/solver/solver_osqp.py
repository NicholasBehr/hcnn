from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from flax import struct

EPS = 1e-12


@struct.dataclass
class OSQPProblem(struct.PyTreeNode):
    ### INITIALIZATION ###
    x0: jax.Array

    ### QUADRATIC COST ###
    # f(x) = 1/2 x.T @ diag(q) @ x + c.T @ x
    cost_quad_diag: jax.Array  # q in cost (vectorized 1d)
    cost_linear: jax.Array  # c in cost (vectorized 1d)

    ### NONLINEAR PARAMETRIC CONSTRAINTS ###
    # l <= Ax <= u
    matrix_vector_product: Callable[[jax.Array], jax.Array]
    vector_matrix_product: Callable[[jax.Array], jax.Array]
    lower_bound: jax.Array  # l in constraint (vectorized 1d)
    upper_bound: jax.Array  # u in constraint (vectorized 1d)


@struct.dataclass
class OSQPParameters(struct.PyTreeNode):
    alpha: float = struct.field(default_factory=lambda: 1.6)
    sigma: float = struct.field(default_factory=lambda: 1e-6)
    rho: float = struct.field(default_factory=lambda: 0.1)
    rho_equality_mult: float = struct.field(default_factory=lambda: 1e3)
    tol_prim: float = struct.field(default_factory=lambda: 1e-2)
    tol_dual: float = struct.field(default_factory=lambda: 1e-2)
    max_iter: int = struct.field(default_factory=lambda: 1e3)
    inner_iterations: int = struct.field(default_factory=lambda: 20)
    tol_infeas_dual: float = struct.field(default_factory=lambda: 1e-6)
    tol_infeas_prim: float = struct.field(default_factory=lambda: 1e-4)


@struct.dataclass
class OSQPVariables(struct.PyTreeNode):
    x_tilde: jax.Array
    z_tilde: jax.Array
    x: jax.Array
    z: jax.Array
    y: jax.Array
    max_primal_residual: jax.Array
    max_dual_residual: jax.Array
    rho_base_mult: jax.Array
    iteration: jax.Array
    is_infeasible: jax.Array

    @staticmethod
    def factory(
        problem: OSQPProblem,
    ):
        x_tilde = problem.x0
        z_tilde = problem.matrix_vector_product(x_tilde)
        x = x_tilde
        z = jnp.clip(z_tilde, problem.lower_bound, problem.upper_bound)
        y = jnp.zeros_like(z)
        # Ax - z
        primal_residuals = z_tilde - z
        # Qx + c + A.Ty
        dual_residuals = problem.cost_quad_diag * x + problem.cost_linear
        rho_base_mult = jnp.array(1.0)
        iteration = jnp.array(0)

        return OSQPVariables(
            x_tilde=x_tilde,
            z_tilde=z_tilde,
            x=x,
            z=z,
            y=y,
            max_primal_residual=jnp.max(jnp.abs(primal_residuals)),
            max_dual_residual=jnp.max(jnp.abs(dual_residuals)),
            rho_base_mult=rho_base_mult,
            iteration=iteration,
            is_infeasible=jnp.array(False),
        )


def osqp_inner_step(
    variables: OSQPVariables,
    problem: OSQPProblem,
    parameters: OSQPParameters,
    rho_vec: jax.Array,
) -> OSQPVariables:
    """Performs one inner OSQP iteration without residual computation."""

    ## x_tilde ##
    def cg_lhs(v: jax.Array) -> jax.Array:
        # Q v + regularization
        a = (problem.cost_quad_diag) * v
        # sigma I v
        b = parameters.sigma * v
        # J.T R J v
        c = problem.matrix_vector_product(v)
        c = rho_vec * c
        c = problem.vector_matrix_product(c)
        return a + b + c

    def approx_inv(v: jax.Array) -> jax.Array:
        return v / (
            problem.cost_quad_diag + parameters.sigma + jnp.mean(rho_vec)
        )

    cg_rhs = (
        parameters.sigma * variables.x
        - problem.cost_linear
        + problem.vector_matrix_product(rho_vec * variables.z - variables.y)
    )

    new_x_tilde, _ = jax.scipy.sparse.linalg.cg(
        A=cg_lhs, b=cg_rhs, x0=variables.x_tilde, M=approx_inv
    )

    ## z_tilde ##
    new_z_tilde = problem.matrix_vector_product(new_x_tilde)

    ## x ##
    new_x = (
        parameters.alpha * new_x_tilde + (1 - parameters.alpha) * variables.x
    )

    ## z ##
    inner = (
        parameters.alpha * new_z_tilde
        + (1 - parameters.alpha) * variables.z
        + variables.y / rho_vec
    )
    new_z = jnp.clip(inner, problem.lower_bound, problem.upper_bound)

    ## y ##
    new_y = variables.y + rho_vec * (
        parameters.alpha * new_z_tilde
        + (1 - parameters.alpha) * variables.z
        - new_z
    )

    return variables.replace(
        x_tilde=new_x_tilde,
        z_tilde=new_z_tilde,
        x=new_x,
        z=new_z,
        y=new_y,
    )


def osqp_step(
    variables: OSQPVariables,
    problem: OSQPProblem,
    parameters: OSQPParameters,
) -> OSQPVariables:
    ## rho ##
    rho_inequality = parameters.rho * variables.rho_base_mult
    rho_equality = rho_inequality * parameters.rho_equality_mult
    rho_vec = jnp.where(
        problem.lower_bound == problem.upper_bound,
        rho_equality,
        rho_inequality,
    )

    ## Inner iterations using scan ##
    def scan_fn(carry, _):
        return osqp_inner_step(carry, problem, parameters, rho_vec), None

    variables_after_inner, _ = jax.lax.scan(
        scan_fn, variables, None, length=parameters.inner_iterations
    )

    ## primal_residuals ##
    new_primal_residuals = (
        problem.matrix_vector_product(variables_after_inner.x)
        - variables_after_inner.z
    )

    ## dual_residuals ##
    new_dual_residuals = (
        problem.cost_quad_diag * variables_after_inner.x
        + problem.cost_linear
        + problem.vector_matrix_product(variables_after_inner.y)
    )

    ## rho_multiplier ##
    mu_sqr = 10**2
    tau_incr = 2
    tau_decr = 2

    # Normalize residuals by convergence tolerances
    normalized_primal_residuals = new_primal_residuals / parameters.tol_prim
    normalized_dual_residuals = new_dual_residuals / parameters.tol_dual

    prim_norm_sqr = jnp.dot(
        normalized_primal_residuals, normalized_primal_residuals
    )
    dual_norm_sqr = jnp.dot(
        normalized_dual_residuals, normalized_dual_residuals
    )

    cond_increase = prim_norm_sqr > mu_sqr * dual_norm_sqr
    cond_decrease = dual_norm_sqr > mu_sqr * prim_norm_sqr
    cond_nochange = ~(cond_increase | cond_decrease)

    dtype = variables.rho_base_mult.dtype
    new_base_mult = (
        tau_incr * variables.rho_base_mult * cond_increase.astype(dtype)
        + (variables.rho_base_mult / tau_decr) * cond_decrease.astype(dtype)
        + variables.rho_base_mult * cond_nochange.astype(dtype)
    )

    ## iteration ##
    new_iteration = variables.iteration + parameters.inner_iterations

    ## Infeasibility detection (following OSQP spec) ##
    # For primal infeasibility detection:
    # Check if ||A^T δy||_∞ <= ε_pinf ||δy||_∞
    # and u^T (δy)+ + l^T (δy)- <= ε_pinf ||δy||_∞
    # where δy = y / ||y||_∞ (normalized y)

    y_norm_inf = jnp.max(jnp.abs(variables_after_inner.y))

    # Avoid division by zero
    y_normalized = jnp.where(
        y_norm_inf > EPS,
        variables_after_inner.y / y_norm_inf,
        variables_after_inner.y,
    )

    # ||A^T δy||_∞
    aty_norm_inf = jnp.max(jnp.abs(problem.vector_matrix_product(y_normalized)))

    # u^T (δy)+ + l^T (δy)-
    y_pos = jnp.maximum(y_normalized, 0)
    y_neg = jnp.minimum(y_normalized, 0)
    primal_infeas_obj = jnp.dot(problem.upper_bound, y_pos) + jnp.dot(
        problem.lower_bound, y_neg
    )

    primal_infeasible = jnp.logical_and(
        aty_norm_inf <= parameters.tol_infeas_dual,
        primal_infeas_obj <= -parameters.tol_infeas_prim,
    )

    # For dual infeasibility detection:
    # Check if ||P δx||_∞ <= ε_dinf ||δx||_∞ and q^T δx <= -ε_dinf ||δx||_∞
    # where δx = x / ||x||_∞ (normalized x) and constraints are satisfied
    x_norm_inf = jnp.max(jnp.abs(variables_after_inner.x))

    # Avoid division by zero
    x_normalized = jnp.where(
        x_norm_inf > EPS,
        variables_after_inner.x / x_norm_inf,
        variables_after_inner.x,
    )

    # ||P δx||_∞ where P = diag(cost_quad_diag)
    px_norm_inf = jnp.max(jnp.abs(problem.cost_quad_diag * x_normalized))

    # q^T δx
    qtx = jnp.dot(problem.cost_linear, x_normalized)

    # Check constraint satisfaction for dual infeasibility
    # For each constraint i: if l_i = -∞, then (Aδx)_i >= 0
    #                       if u_i = +∞, then (Aδx)_i <= 0
    #                       if l_i = u_i, then (Aδx)_i = 0
    ax_normalized = problem.matrix_vector_product(x_normalized)

    # Create masks for different constraint types
    lower_inf = jnp.isinf(problem.lower_bound) & (problem.lower_bound < 0)
    upper_inf = jnp.isinf(problem.upper_bound) & (problem.upper_bound > 0)
    eps = 1e-12
    equality = jnp.abs(problem.upper_bound - problem.lower_bound) < eps

    # Check constraint satisfaction
    lower_satisfied = jnp.logical_or(
        ~lower_inf, ax_normalized >= -parameters.tol_infeas_prim
    )
    upper_satisfied = jnp.logical_or(
        ~upper_inf, ax_normalized <= parameters.tol_infeas_prim
    )
    equality_satisfied = jnp.logical_or(
        ~equality, jnp.abs(ax_normalized) <= parameters.tol_infeas_prim
    )

    constraints_satisfied = jnp.all(
        lower_satisfied & upper_satisfied & equality_satisfied
    )

    dual_infeasible = jnp.logical_and(
        jnp.logical_and(
            px_norm_inf <= parameters.tol_infeas_dual,
            qtx <= -parameters.tol_infeas_prim,
        ),
        constraints_satisfied,
    )

    is_infeasible = jnp.logical_or(primal_infeasible, dual_infeasible)

    return OSQPVariables(
        x_tilde=variables_after_inner.x_tilde,
        z_tilde=variables_after_inner.z_tilde,
        x=variables_after_inner.x,
        z=variables_after_inner.z,
        y=variables_after_inner.y,
        max_primal_residual=jnp.max(jnp.abs(new_primal_residuals)),
        max_dual_residual=jnp.max(jnp.abs(new_dual_residuals)),
        rho_base_mult=new_base_mult,
        iteration=new_iteration,
        is_infeasible=is_infeasible,
    )


def osqp_not_converged(
    variables: OSQPVariables,
    parameters: OSQPParameters,
) -> jax.Array:
    # Return early if problem is detected as infeasible
    infeasible = variables.is_infeasible

    # check if primal or dual residuals are above the tolerance
    primal = variables.max_primal_residual >= parameters.tol_prim
    dual = variables.max_dual_residual >= parameters.tol_dual
    residual = jnp.logical_or(primal, dual)

    # check if maximum number of iterations has not been reached
    iteration = variables.iteration < parameters.max_iter

    # declare not converged if residuals are high AND iterations remain
    # BUT return converged (False) if infeasible is detected
    not_converged_by_residual = jnp.logical_and(residual, iteration)

    # Return True (not converged) only if not infeasible and residuals/iter
    return jnp.logical_and(
        jnp.logical_not(infeasible), not_converged_by_residual
    )


def osqp_solve(
    problem: OSQPProblem,
    parameters: OSQPParameters,
) -> OSQPVariables:
    osqp_variables = jax.lax.while_loop(
        cond_fun=partial(
            osqp_not_converged,
            parameters=parameters,
        ),
        body_fun=partial(
            osqp_step,
            problem=problem,
            parameters=parameters,
        ),
        init_val=OSQPVariables.factory(
            problem=problem,
        ),
    )

    # jax.debug.print(
    #     "osqp prim {}, dual {}, iter {}, mult {}, infeas {}",
    #     osqp_variables.max_primal_residual,
    #     osqp_variables.max_dual_residual,
    #     osqp_variables.iteration,
    #     osqp_variables.rho_base_mult,
    #     osqp_variables.is_infeasible,
    # )

    return osqp_variables
