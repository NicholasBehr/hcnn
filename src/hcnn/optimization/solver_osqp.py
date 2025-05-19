from functools import partial
from typing import Self

import jax
import jax.numpy as jnp
from flax import nnx, struct

from .optimization import (
    OptimizationProblem,
    OptimizationSolverBase,
)


@struct.dataclass
class OptimizationSolverOSQP(OptimizationSolverBase):
    max_iter: int = 500
    tol: float = 1e-4

    class Variables(struct.PyTreeNode):
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

        @staticmethod
        def factory(problem: OptimizationProblem) -> "OptimizationSolverOSQP.Variables":
            x = problem.x0
            z = problem.constraint_fn(x)
            return OptimizationSolverOSQP.Variables(
                # primal
                x=x,
                x_tilde=x,
                z=z,
                z_tilde=z,
                primal_residuals=jnp.zeros_like(z),
                # dual
                y=jnp.zeros_like(z),
                dual_residuals=problem.cost_quad_diag * x + problem.cost_linear,
                # parameters
                alpha=jnp.array(1.6, dtype=float),
                sigma=jnp.array(1e-6, dtype=float),
                rho_vec=0.3 * jnp.where(problem.lower_bound == problem.upper_bound, 1e3, 1.0),
                iter_counter=jnp.array(0, dtype=int),
            )

        def update_primal_dual(self, problem: OptimizationProblem, tol) -> Self:
            ## form linearization around x_tilde
            h, h_jvp = jax.linearize(problem.constraint_fn, self.x_tilde)
            _, h_vjp = jax.vjp(problem.constraint_fn, self.x_tilde)

            ## x_tilde ##
            def cg_preconditioner(v: jax.Array) -> jax.Array:
                # (Q + sigma I) @ v
                return (problem.cost_quad_diag + self.sigma) * v

            def cg_lhs(v: jax.Array) -> jax.Array:
                # Q v
                a = problem.cost_quad_diag * v
                # sigma I v
                b = self.sigma * v
                # J.T R J v
                c = h_jvp(v)
                c = self.rho_vec * c
                c = h_vjp(c)[0]
                return a + b + c

            zhjx = self.z - h + h_jvp(self.x_tilde)
            r_zhjx_y = self.rho_vec * zhjx - self.y
            cg_rhs = self.sigma * self.x - problem.cost_linear + h_vjp(r_zhjx_y)[0]

            new_x_tilde, _ = jax.scipy.sparse.linalg.cg(
                A=cg_lhs, b=cg_rhs, x0=self.x_tilde, M=cg_preconditioner, tol=tol
            )

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
            ## form linearization around x
            h, h_vjp = jax.vjp(problem.constraint_fn, self.x)

            primal_residuals = h - self.z
            dual_residuals = (
                problem.cost_quad_diag * self.x + problem.cost_linear + h_vjp(self.y)[0]
            )

            return self.replace(
                primal_residuals=primal_residuals,
                dual_residuals=dual_residuals,
            )

        def update_rho(self) -> Self:
            mu_sqr = 10**2
            tau_incr = 2
            tau_decr = 2

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
        primal_check = jnp.max(jnp.abs(variables.primal_residuals)) >= self.tol
        dual_check = jnp.max(jnp.abs(variables.dual_residuals)) >= self.tol
        not_converged_check = jnp.logical_or(primal_check, dual_check)

        # AND do we have iterations left?
        iter_remaining_check = variables.iter_counter < self.max_iter

        return jnp.logical_and(not_converged_check, iter_remaining_check)

    def body_fun(
        self,
        variables: "OptimizationSolverOSQP.Variables",
        problem: OptimizationProblem,
    ) -> Variables:
        return (
            variables.replace(iter_counter=variables.iter_counter + 1)
            .update_rho()
            .update_primal_dual(problem=problem, tol=self.tol / 10)
            .update_residuals(problem=problem)
        )

    def _solve_variables(self, problem: OptimizationProblem) -> Variables:
        variables = OptimizationSolverOSQP.Variables.factory(problem)
        variables = jax.lax.while_loop(
            cond_fun=self.cond_fun,
            body_fun=partial(self.body_fun, problem=problem),
            init_val=variables,
        )
        return variables

    def _solve(self, problem: OptimizationProblem):
        return self._solve_variables(problem).x

    def _solve_fwd(self, problem: OptimizationProblem):
        variables = self._solve_variables(problem)
        return variables.x, (variables.x, variables.y, problem)

    @partial(nnx.jit, static_argnums=0)
    def _solve_bwd(self, res, g):
        x_star, y_star, problem = res
        (state_updates_g,), out_g = g

        ## Precompute JVP, VJP and HVP for the constraint function h(x) = problem.constraint_fn_x(x)
        # h_jvp computes v -> dh/dx @ v
        _, h_jvp = jax.linearize(problem.constraint_fn, x_star)
        # h_vjp computes w -> w^T @ dh/dx
        _, h_vjp = jax.vjp(problem.constraint_fn, x_star)
        # h_hvp computes y.T @ d^2h/dx^2 @ v
        _, h_hvp = jax.linearize(
            jax.grad(lambda x: jnp.dot(y_star, problem.constraint_fn(x))), x_star
        )
        abs_y_star = jnp.abs(y_star)

        ## Define the lhs function using the precomputed components
        def lhs(u):
            u1, u3 = u

            # Q @ u1
            term1 = problem.cost_quad_diag * u1
            # y.T @ d^2h/dx^2 @ u1
            term2 = h_hvp(u1)
            # (dh/dx).T @ D1 @ u3
            term3 = h_vjp(abs_y_star * u3)[0]

            v1 = term1 + term2 + term3

            # dh/dx u1
            v3 = h_jvp(u1) + 1e-12 * u3

            return (v1, v3)

        rhs = (out_g, jnp.zeros_like(y_star))

        (u1, u3), _ = jax.scipy.sparse.linalg.gmres(A=lhs, b=rhs, x0=rhs)
        u2 = abs_y_star * u3

        ## explicit vjp (step 2)
        graph, state = nnx.split(problem)

        def param_constraint_fn(x, state):
            param_problem = nnx.merge(graph, state)
            return param_problem.constraint_fn(x)

        constraint_params_term1 = jax.grad(
            lambda phi_arg: jnp.dot(
                jax.jvp(lambda x_inner: param_constraint_fn(x_inner, phi_arg), (x_star,), (-u1,))[
                    1
                ],
                y_star,
            )
        )(state)

        constraint_params_term2 = jax.vjp(
            lambda phi_arg: param_constraint_fn(x_star, phi_arg),
            state,
        )[1](-u2)[0]

        grad_constraint_params = jax.tree_util.tree_map(
            jnp.add,
            constraint_params_term1,
            constraint_params_term2,
        )

        m_g = jax.tree.map(lambda x: x, state_updates_g)  # create copy
        m_g["cost_quad_diag"].value += -u1 * x_star
        m_g["cost_linear"].value += -u1
        m_g["lower_bound"].value += jnp.where(y_star < -self.tol, -u3 * y_star, 0)
        m_g["upper_bound"].value += jnp.where(y_star > +self.tol, +u3 * y_star, 0)
        for k, v in grad_constraint_params.items():
            if isinstance(v, nnx.VariableState):
                m_g[k].value += v.value
            else:
                m_g[k] += v

        print(m_g)

        return (m_g,)

    _solve_vjp = nnx.custom_vjp(_solve, nondiff_argnums=(0,))
    _solve_vjp.defvjp(_solve_fwd, _solve_bwd)
