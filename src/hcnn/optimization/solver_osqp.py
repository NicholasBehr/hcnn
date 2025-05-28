from functools import partial
from typing import Self, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from .optimization import OptimizationProblemBase, OptimizationSolverBase, PhiType


@struct.dataclass
class OptimizationSolverOSQP(OptimizationSolverBase[PhiType]):
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
        def factory(problem: OptimizationProblemBase, phi: PhiType) -> "OptimizationSolverOSQP.Variables":
            x = problem.get_x0(phi)
            z = problem.constraint_fn(x, phi)
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

        def update_primal_dual(self, problem: OptimizationProblemBase, tol: float, phi: PhiType) -> Self:
            ## form linearization around x_tilde
            h, h_jvp = jax.linearize(lambda x: problem.constraint_fn(x, phi), self.x_tilde)
            _, h_vjp = jax.vjp(lambda x: problem.constraint_fn(x, phi), self.x_tilde)

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
            new_y = self.y + self.rho_vec * (self.alpha * new_z_tilde + (1 - self.alpha) * self.z - new_z)

            return self.replace(
                x_tilde=new_x_tilde,
                z_tilde=new_z_tilde,
                x=new_x,
                z=new_z,
                y=new_y,
            )

        def update_residuals(self, problem: OptimizationProblemBase, phi: PhiType) -> Self:
            ## form linearization around x
            h, h_vjp = jax.vjp(lambda x: problem.constraint_fn(x, phi), self.x)

            primal_residuals = h - self.z
            dual_residuals = problem.cost_quad_diag * self.x + problem.cost_linear + h_vjp(self.y)[0]

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
        problem: OptimizationProblemBase,
        phi: PhiType,
    ) -> Variables:
        return (
            variables.replace(iter_counter=variables.iter_counter + 1)
            .update_rho()
            .update_primal_dual(problem=problem, tol=self.tol / 10, phi=phi)
            .update_residuals(problem=problem, phi=phi)
        )

    def _solve_variables(self, problem: OptimizationProblemBase, phi: PhiType) -> Variables:
        variables = OptimizationSolverOSQP.Variables.factory(problem, phi)
        variables = jax.lax.while_loop(
            cond_fun=self.cond_fun,
            body_fun=partial(self.body_fun, problem=problem, phi=phi),
            init_val=variables,
        )
        return variables

    def _solve_for_unrolling(self, problem: OptimizationProblemBase, phi: PhiType):
        variables = OptimizationSolverOSQP.Variables.factory(problem, phi)
        for _ in range(35):
            variables = self.body_fun(variables, problem, phi)
        return variables.x

    def _solve(self, problem: OptimizationProblemBase, phi: PhiType):
        return self._solve_variables(problem, phi).x

    def _solve_fwd(self, problem: OptimizationProblemBase, phi: PhiType):
        variables = self._solve_variables(problem, phi)
        return variables.x, (variables.x, variables.z, variables.y, problem, phi)

    def _solve_bwd(
        self,
        res: Tuple[jax.Array, jax.Array, OptimizationProblemBase, PhiType],
        g: jax.Array,
    ):
        x_star, z_star, y_star, problem, phi = res
        y_plus = jnp.maximum(y_star, 0)
        y_minus = jnp.minimum(y_star, 0)
        v1 = g

        # TODO REDO STUPID (BUT SAFE?) FULL JACOB IMPLEMENTATION AFTER BUGFIX
        Ax_fn = jax.jacrev(lambda x: problem.constraint_fn(x, phi))
        Ax = Ax_fn(x_star)

        # jax.debug.print("Ax {}, \nyAxx {}, \ny_star{}", Ax, yAxx, y_star)

        n = x_star.shape[0]
        m = z_star.shape[0]

        lhs = jnp.vstack(
            [
                jnp.hstack([jnp.diag(problem.cost_quad_diag), jnp.zeros((n, m)), Ax.T, Ax.T]),
                jnp.hstack([Ax, -jnp.eye(m), jnp.zeros((m, m)), jnp.zeros((m, m))]),
                jnp.hstack(
                    [jnp.zeros((m, n)), jnp.diag(y_plus), jnp.diag(z_star - problem.upper_bound), jnp.zeros((m, m))]
                ),
                jnp.hstack(
                    [jnp.zeros((m, n)), jnp.diag(y_minus), jnp.zeros((m, m)), jnp.diag(z_star - problem.lower_bound)]
                ),
            ]
        )
        rhs = -jnp.concatenate([v1, jnp.zeros((3 * m,))])
        # jax.debug.print("lhs: \n{}, \nrhs:\n{}\n", lhs, rhs)
        out, _ = jax.scipy.sparse.linalg.gmres(A=lhs.T, b=rhs)
        # jax.debug.print("out: {}, lhs@out: {}", out, lhs @ out)
        dx = out[:n]
        dz = out[n : n + m]
        dyp = out[n + m : n + 2 * m]
        dym = out[n + 2 * m :]
        jax.debug.print("dx {}, dz {}, dyp {}, dym {}", dx, dz, dyp, dym)

        ## explicit vjp (step 2)
        g_phi_1 = jax.grad(
            lambda phi_arg: jnp.dot(
                jax.jvp(lambda x_inner: problem.constraint_fn(x_inner, phi_arg), (x_star,), (dx,))[1],
                y_star,
            )
        )(phi)
        g_phi_2 = jax.vjp(lambda phi_arg: problem.constraint_fn(x_star, phi_arg), phi)[1](dz)[0]
        g_phi = jax.tree_util.tree_map(jnp.add, g_phi_1, g_phi_2)

        jax.debug.print("phi1: {}, \nphi2: {}", g_phi_1, g_phi_2)

        g_problem = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), problem)  # copy
        g_problem = g_problem.replace(
            cost_quad_diag=dx * x_star,
            cost_linear=dx,
            upper_bound=dyp @ jnp.diag(-y_plus),
            lower_bound=dym @ jnp.diag(-y_minus),
        )

        return (g_problem, g_phi)

    _solve_vjp = jax.custom_vjp(_solve, nondiff_argnums=(0,))
    _solve_vjp.defvjp(_solve_fwd, _solve_bwd)
