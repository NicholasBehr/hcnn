from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from flax import struct


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
    tol: float = struct.field(default_factory=lambda: 1e-4)
    max_iter: int = struct.field(default_factory=lambda: 5_000)
    inner_iterations: int = struct.field(default_factory=lambda: 50)


@struct.dataclass
class OSQPVariables(struct.PyTreeNode):
    x_tilde: jax.Array
    z_tilde: jax.Array
    x: jax.Array
    z: jax.Array
    y: jax.Array
    primal_residuals: jax.Array
    dual_residuals: jax.Array
    rho_base_mult: jax.Array
    iteration: jax.Array

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
        rho_residual_mult = jnp.array(1)
        iteration = jnp.array(0)

        return OSQPVariables(
            x_tilde=x_tilde,
            z_tilde=z_tilde,
            x=x,
            z=z,
            y=y,
            primal_residuals=primal_residuals,
            dual_residuals=dual_residuals,
            rho_base_mult=rho_residual_mult,
            iteration=iteration,
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

    def cg_lhs_approx_inv(v: jax.Array) -> jax.Array:
        return v / (problem.cost_quad_diag + parameters.sigma)

    cg_rhs = (
        parameters.sigma * variables.x
        - problem.cost_linear
        + problem.vector_matrix_product(rho_vec * variables.z - variables.y)
    )

    new_x_tilde, _ = jax.scipy.sparse.linalg.cg(
        A=cg_lhs, b=cg_rhs, x0=variables.x_tilde, M=cg_lhs_approx_inv
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

    prim_norm_sqr = jnp.dot(new_primal_residuals, new_primal_residuals)
    dual_norm_sqr = jnp.dot(new_dual_residuals, new_dual_residuals)

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

    return OSQPVariables(
        x_tilde=variables_after_inner.x_tilde,
        z_tilde=variables_after_inner.z_tilde,
        x=variables_after_inner.x,
        z=variables_after_inner.z,
        y=variables_after_inner.y,
        primal_residuals=new_primal_residuals,
        dual_residuals=new_dual_residuals,
        rho_base_mult=new_base_mult,
        iteration=new_iteration,
    )


def osqp_not_converged(
    variables: OSQPVariables,
    parameters: OSQPParameters,
) -> jax.Array:
    # check if primal or dual residuals are above the tolerance
    primal = jnp.max(jnp.abs(variables.primal_residuals)) >= parameters.tol
    dual = jnp.max(jnp.abs(variables.dual_residuals)) >= parameters.tol
    residual = jnp.logical_or(primal, dual)

    # check if maximum number of iterations has not been reached
    iteration = variables.iteration < parameters.max_iter

    # declare not converged if both conditions are true
    return jnp.logical_and(residual, iteration)


def osqp_solve(
    problem: OSQPProblem,
    parameters: OSQPParameters,
) -> OSQPVariables:
    return jax.lax.while_loop(
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
