from functools import partial
from typing import Self

import jax
import jax.numpy as jnp
from flax import struct

from hcnn.optimization.optimization import (
    OptimizationProblem,
    OptimizationSolver,
)


@struct.dataclass
class SolverQSQP(OptimizationSolver):
    @struct.dataclass
    class Variables:
        # linearization
        x_lin: jax.Array
        # primal
        x: jax.Array
        x_tilde: jax.Array
        z: jax.Array
        z_tilde: jax.Array
        primal_residuals: jax.Array
        # dual
        y: jax.Array
        dual_residuals: jax.Array
        # parameters
        alpha: jax.Array
        sigma: jax.Array
        rho_vec: jax.Array
        max_major_iter: int

        @staticmethod
        def factory(problem: OptimizationProblem) -> Self:
            x = problem.x0
            z = problem.constraints(x)
            return SolverQSQP.Variables(
                # linearization
                x_lin=x,
                # primal
                x=x,
                x_tilde=x,
                z=z,
                z_tilde=z,
                primal_residuals=jnp.zeros_like(z),
                # dual
                y=jnp.zeros_like(z),
                dual_residuals=problem.cost_qvp(x) + problem.cost_linear,
                # parameters
                alpha=jnp.atleast_1d(1.6),
                sigma=jnp.atleast_1d(1e-6),
                rho_vec=0.1 * jnp.where(problem.lower_bound == problem.upper_bound, 1e3, 1.0),
                max_major_iter=15,
            )

        def update_linearization(self) -> Self:
            return self.replace(x_lin=self.x_tilde)

        def update_primal_dual(self, problem: OptimizationProblem) -> Self:
            ## form linearization
            h, h_jvp = jax.linearize(problem.constraints, self.x_lin)
            _, h_vjp = jax.vjp(problem.constraints, self.x_lin)

            ## x_tilde ##
            def cg_preconditioner(v: jax.Array) -> jax.Array:
                # Qv
                a = problem.cost_qvp(v)
                # sigmaIv
                b = self.sigma * v
                return a + b

            def cg_lhs(v: jax.Array) -> jax.Array:
                # Qv
                a = problem.cost_qvp(v)
                # sigmaIv
                b = self.sigma * v
                # JRJv
                c = h_jvp(v)
                c = self.rho_vec * c
                c = h_vjp(c)[0]
                return a + b + c

            zhjx = self.z - h + h_jvp(self.x_lin)
            r_zhjx_y = self.rho_vec * zhjx - self.y
            cg_rhs = self.sigma * self.x - problem.cost_linear + h_vjp(r_zhjx_y)[0]

            new_x_tilde = jax.scipy.sparse.linalg.cg(
                A=cg_lhs, b=cg_rhs, x0=self.x_tilde, M=cg_preconditioner
            )[0]

            ## z_tilde ##
            new_z_tilde = h + h_jvp(new_x_tilde - self.x_lin)

            ## x ##
            new_x = self.alpha * new_x_tilde + (1 - self.alpha) * self.x

            ## z ##
            inner = self.alpha * new_z_tilde + (1 - self.alpha) * self.z + self.y / self.rho_vec
            new_z = jnp.clip(inner, problem.lower_bound, problem.upper_bound)

            ## y ##
            new_y = self.y + self.rho_vec * (
                self.alpha * new_z_tilde + (1 - self.alpha) * self.z - new_z
            )

            ## residuals ##
            primal_residuals = h + h_jvp(self.x - self.x_lin) - self.z
            dual_residuals = problem.cost_qvp(self.x) + problem.cost_linear + h_vjp(self.y)[0]

            return self.replace(
                x_tilde=new_x_tilde,
                z_tilde=new_z_tilde,
                x=new_x,
                z=new_z,
                y=new_y,
                primal_residuals=primal_residuals,
                dual_residuals=dual_residuals,
            )

        def update_rho(self) -> Self:
            mu_sqr = 10**2
            tau_incr = 2.0
            tau_decr = 2.0

            prim_norm_sqr = jnp.dot(self.primal_residuals, self.primal_residuals)
            dual_norm_sqr = jnp.dot(self.dual_residuals, self.dual_residuals)

            cond_increase = prim_norm_sqr > mu_sqr * dual_norm_sqr
            cond_decrease = dual_norm_sqr > mu_sqr * prim_norm_sqr
            cond_nochange = ~(cond_increase | cond_decrease)

            dtype = self.rho_vec.dtype

            rho_vec = (
                tau_incr * self.rho_vec * cond_increase.astype(dtype)
                + (self.rho_vec / tau_decr) * cond_decrease.astype(dtype)
                + self.rho_vec * cond_nochange.astype(dtype)
            )

            return self.replace(rho_vec=rho_vec)

    @staticmethod
    def minor_iteration(
        i, variables: "SolverQSQP.Variables", problem: OptimizationProblem
    ) -> "SolverQSQP.Variables":
        return variables.update_primal_dual(problem).update_rho()

    @staticmethod
    def major_cond_fun(variables: "SolverQSQP.Variables", atol: float) -> bool:
        # are we not yet converged?
        primal_check = jnp.max(jnp.abs(variables.primal_residuals)) > atol
        dual_check = jnp.max(jnp.abs(variables.dual_residuals)) > atol
        not_converged_check = jnp.logical_or(primal_check, dual_check)
        # and do we have major iters left?
        max_major_iter_check = variables.max_major_iter > 0
        # return true if both and we should enter a new major iteration
        return jnp.logical_and(not_converged_check, max_major_iter_check)

    @staticmethod
    def major_iteration(
        variables: "SolverQSQP.Variables", problem: OptimizationProblem
    ) -> "SolverQSQP.Variables":
        variables = variables.update_linearization()
        variables = jax.lax.fori_loop(
            lower=0,
            upper=1,
            body_fun=partial(SolverQSQP.minor_iteration, problem=problem),
            init_val=variables,
            unroll=True,
        )
        variables = variables.replace(max_major_iter=variables.max_major_iter - 1)
        return variables

    @staticmethod
    def solve(problem: OptimizationProblem) -> jax.Array:
        variables = SolverQSQP.Variables.factory(problem)
        variables = jax.lax.while_loop(
            cond_fun=partial(SolverQSQP.major_cond_fun, atol=1e-5),
            body_fun=partial(SolverQSQP.major_iteration, problem=problem),
            init_val=variables,
        )

        return variables.x
