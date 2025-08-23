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
    tol: float = struct.field(default_factory=lambda: 1e-4)
    tau: float = struct.field(default_factory=lambda: 1e-12)
    epsilon: float = struct.field(default_factory=lambda: 1e-12)
    max_iter_lin: int = struct.field(default_factory=lambda: 5_000)
    max_iter_nonlin: int = struct.field(default_factory=lambda: 500)


@struct.dataclass
class SQPVariables(struct.PyTreeNode):
    x: jax.Array
    z: jax.Array
    y: jax.Array
    r_x: jax.Array
    r_y: jax.Array
    q_plus_s: jax.Array
    nonlin_iter: jax.Array
    sumlin_iter: jax.Array
    sum_rho_base_mult: jax.Array

    @staticmethod
    def factory(problem: SQPProblemBase):
        x = problem.x0
        h = problem.constraint_fn(x)
        z = jnp.clip(h, problem.lower_bound, problem.upper_bound)
        y = jnp.zeros_like(z)

        # Qx + c + A.T y
        r_x = problem.cost_quad_diag * x + problem.cost_linear
        # h(x) - z
        r_y = h - z
        q_plus_s = jnp.zeros_like(problem.cost_quad_diag)

        return SQPVariables(
            x=x,
            z=z,
            y=y,
            r_x=r_x,
            r_y=r_y,
            q_plus_s=q_plus_s,
            nonlin_iter=jnp.array(0),
            sumlin_iter=jnp.array(0),
            sum_rho_base_mult=jnp.array(0),
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
        tol=parameters.tol,
        max_iter=parameters.max_iter_lin,
    )
    osqp_variables = osqp_solve(osqp_problem, osqp_parameters)

    # update variables
    new_x = variables.x + osqp_variables.x
    new_z = offset + osqp_variables.z
    new_y = osqp_variables.y

    ## residuals ##
    new_h, new_h_vjp = jax.vjp(problem.constraint_fn, new_x)
    new_r_x = (
        jnp.maximum(problem.cost_quad_diag, parameters.epsilon) * new_x
        + problem.cost_linear
        + new_h_vjp(new_y)[0]
    )
    new_r_y = new_h - new_z

    ## iteration ##
    new_nonlin_iter = variables.nonlin_iter + 1
    new_sumlin_iter = variables.sumlin_iter + osqp_variables.iteration

    ## log multiplier ##
    new_sum_rho_base_mult = (
        variables.sum_rho_base_mult + osqp_variables.rho_base_mult
    )

    return SQPVariables(
        x=new_x,
        z=new_z,
        y=new_y,
        r_x=new_r_x,
        r_y=new_r_y,
        q_plus_s=new_q_plus_s,
        nonlin_iter=new_nonlin_iter,
        sumlin_iter=new_sumlin_iter,
        sum_rho_base_mult=new_sum_rho_base_mult,
    )


def sqp_not_converged(
    variables: SQPVariables,
    parameters: SQPParameters,
) -> jax.Array:
    # check if primal or dual residuals are above the tolerance
    res_x = jnp.max(jnp.abs(variables.r_x)) >= parameters.tol
    res_y = jnp.max(jnp.abs(variables.r_y)) >= parameters.tol
    residual = jnp.logical_or(res_x, res_y)

    # check if maximum number of iterations has not been reached
    iteration = variables.nonlin_iter < parameters.max_iter_nonlin

    # declare not converged if residuals high AND iterations not exceeded
    return jnp.logical_and(residual, iteration)


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
        init_val=SQPVariables.factory(problem=problem),
    )
    sqp_info = SQPInfo(
        primal_residual=jnp.abs(variables.r_y),
        dual_residual=jnp.abs(variables.r_x),
        iterations_lin=variables.sumlin_iter / variables.nonlin_iter,
        iterations_nonlin=variables.nonlin_iter,
        rho_base_mult=variables.sum_rho_base_mult / variables.nonlin_iter,
        converged_flag=jnp.logical_and(
            jnp.max(jnp.abs(variables.r_x)) < parameters.tol,
            jnp.max(jnp.abs(variables.r_y)) < parameters.tol,
        ),
    )
    return variables, sqp_info


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def sqp_solve(
    problem: SQPProblemBase,
    parameters: SQPParameters,
) -> tuple[jax.Array, SQPInfo]:
    variables, sqp_info = sqp_solve_variables(
        problem=problem,
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
    variables, sqp_info = sqp_solve_variables(
        problem=problem,
        parameters=parameters,
    )

    return (
        (variables.x, sqp_info),
        (
            variables.x,  # primal solution
            variables.y,  # dual variables
            variables.q_plus_s,  # diagonal Hessian approx
            sqp_info.converged_flag,  # converged flag
            problem,  # problem data (differentiable arg structure)
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

        frac_upper = y_plus / (parameters.tau + problem.upper_bound - h)
        frac_lower = y_minus / (parameters.tau + h - problem.lower_bound)
        frac_sum = frac_upper + frac_lower

        def schur_lhs(u1: jax.Array) -> jax.Array:
            return q_plus_s * u1 + h_vjp(frac_sum * h_jvp(u1))[0]

        u1, _ = jax.scipy.sparse.linalg.cg(A=schur_lhs, b=v1)

        jvp_u1 = h_jvp(u1)
        u2 = jvp_u1 / (parameters.tau + problem.upper_bound - h)
        u3 = jvp_u1 / (problem.lower_bound - parameters.tau - h)

        ## Explicit VJP (step 2)`
        grad_problem = jax.tree.map(jnp.zeros_like, problem)  # empty copy

        g_q = -x_star * u1
        g_c = -u1
        g_u = y_plus * u2
        g_l = -y_minus * u3

        ## phi grad
        g_phi_d_1 = jax.grad(
            lambda phi_arg: jnp.dot(
                jax.jvp(
                    lambda x_inner: problem.constraint_fn_phi(x_inner, phi_arg),
                    (x_star,),
                    (-u1,),
                )[1],
                y_star,
            ),
        )(problem.phi)

        g_phi_d_2 = jax.vjp(
            lambda phi_arg: problem.constraint_fn_phi(x_star, phi_arg),
            problem.phi,
        )[1](-(u2 * y_plus - u3 * y_minus))[0]
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
