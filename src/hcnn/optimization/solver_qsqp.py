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
    max_iter: int = 200
    atol: float = 1e-5

    @struct.dataclass
    class Variables:
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
        iter_counter: jax.Array

        def replace(self, **kwargs) -> Self:
            """
            Placeholder for typing, will get overridden by @struct.dataclass
            """
            return self

        @staticmethod
        def factory(problem: OptimizationProblem) -> "SolverQSQP.Variables":
            x = problem.x0
            z = problem.constraints(x)
            return SolverQSQP.Variables(
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
                alpha=jnp.array(1.6, dtype=float),
                sigma=jnp.array(1e-6, dtype=float),
                rho_vec=0.1 * jnp.where(problem.lower_bound == problem.upper_bound, 1e3, 1.0),
                iter_counter=jnp.array(0, dtype=int),
            )

        def update_primal_dual(self, problem: OptimizationProblem, atol: float) -> Self:
            ## form linearization
            h, h_jvp = jax.linearize(problem.constraints, self.x_tilde)
            _, h_vjp = jax.vjp(problem.constraints, self.x_tilde)

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

            zhjx = self.z - h + h_jvp(self.x_tilde)
            r_zhjx_y = self.rho_vec * zhjx - self.y
            cg_rhs = self.sigma * self.x - problem.cost_linear + h_vjp(r_zhjx_y)[0]

            new_x_tilde = jax.scipy.sparse.linalg.cg(
                A=cg_lhs,
                b=cg_rhs,
                x0=self.x_tilde,
                M=cg_preconditioner,
                atol=atol,
            )[0]

            ## z_tilde ##
            new_z_tilde = h + h_jvp(new_x_tilde - self.x_tilde)

            ## x ##
            new_x = self.alpha * new_x_tilde + (1 - self.alpha) * self.x

            ## z ##
            inner = self.alpha * new_z_tilde + (1 - self.alpha) * self.z + self.y / self.rho_vec
            new_z = jnp.clip(inner, problem.lower_bound, problem.upper_bound)

            ## y ##
            new_y = self.y + self.rho_vec * (
                self.alpha * new_z_tilde + (1 - self.alpha) * self.z - new_z
            )

            return self.replace(
                x_tilde=new_x_tilde,
                z_tilde=new_z_tilde,
                x=new_x,
                z=new_z,
                y=new_y,
            )

        def update_residuals(self, problem: OptimizationProblem) -> Self:
            h, h_vjp = jax.vjp(problem.constraints, self.x)

            primal_residuals = h - self.z
            dual_residuals = problem.cost_qvp(self.x) + problem.cost_linear + h_vjp(self.y)[0]

            return self.replace(
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

            return self.replace(
                rho_vec=rho_vec,
            )

    def cond_fun(self, variables: Variables) -> jax.Array:
        # return true if we should enter a new iteration

        # are we not converged?
        primal_check = jnp.max(jnp.abs(variables.primal_residuals)) >= self.atol
        dual_check = jnp.max(jnp.abs(variables.dual_residuals)) >= self.atol
        x_tilde_check = jnp.max(jnp.abs(variables.x - variables.x_tilde)) >= self.atol
        z_tilde_check = jnp.max(jnp.abs(variables.z - variables.z_tilde)) >= self.atol
        not_converged_check = jnp.logical_or(
            jnp.logical_or(primal_check, dual_check),
            jnp.logical_or(x_tilde_check, z_tilde_check),
        )
        # AND do we have iterations left?
        max_iter_check = variables.iter_counter < self.max_iter

        return jnp.logical_and(not_converged_check, max_iter_check)

    def body_fun(
        self, variables: "SolverQSQP.Variables", problem: OptimizationProblem
    ) -> Variables:
        return (
            variables.replace(iter_counter=variables.iter_counter + 1)
            .update_rho()
            .update_primal_dual(problem=problem, atol=self.atol)
            .update_residuals(problem=problem)
        )

    def _solve_full(self, problem: OptimizationProblem) -> Variables:
        variables = SolverQSQP.Variables.factory(problem)
        variables = jax.lax.while_loop(
            cond_fun=partial(self.cond_fun),
            body_fun=partial(self.body_fun, problem=problem),
            init_val=variables,
        )
        return variables

    def solve(self, problem: OptimizationProblem) -> jax.Array:
        return self._solve_full(problem).x
