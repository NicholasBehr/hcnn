import jax
import jax.numpy as jnp
from flax import struct

from .optimization_base import (
    OptimizationProblemBase,
    OptimizationSolverBase,
    OptimizationVariables,
    PhiTypeDiff,
    PhiTypeNoDiff,
)


@struct.dataclass
class OptimizationSolverOSQP(
    OptimizationSolverBase[PhiTypeDiff, PhiTypeNoDiff]
):
    def update_primal_dual(
        self,
        variables: OptimizationVariables,
        problem: OptimizationProblemBase[PhiTypeDiff, PhiTypeNoDiff],
        phi_d: PhiTypeDiff,
        phi_nd: PhiTypeNoDiff,
    ) -> OptimizationVariables:
        ## form linearization around x_tilde
        h, h_jvp = jax.linearize(
            lambda x: problem.constraint_fn(x, phi_d, phi_nd),
            variables.x_tilde,
        )
        _, h_vjp = jax.vjp(
            lambda x: problem.constraint_fn(x, phi_d, phi_nd),
            variables.x_tilde,
        )

        # form rho vector
        rho_vec = jnp.where(
            problem.lower_bound == problem.upper_bound,
            self.rho_equality,
            self.rho_inequality,
        )

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
            c = rho_vec * c
            c = h_vjp(c)[0]
            return a + b + c

        zhjx = variables.z - h + h_jvp(variables.x_tilde)
        r_zhjx_y = rho_vec * zhjx - variables.y
        cg_rhs = (
            self.sigma * variables.x - problem.cost_linear + h_vjp(r_zhjx_y)[0]
        )

        new_x_tilde, _ = jax.scipy.sparse.linalg.cg(
            A=cg_lhs,
            b=cg_rhs,
            x0=variables.x_tilde,
            M=cg_preconditioner,
            tol=self.tol / 2,
        )

        ## z_tilde ##
        new_z_tilde = h + h_jvp(new_x_tilde - variables.x_tilde)

        ## x ##
        new_x = self.alpha * new_x_tilde + (1 - self.alpha) * variables.x

        ## z ##
        inner = (
            self.alpha * new_z_tilde
            + (1 - self.alpha) * variables.z
            + variables.y / rho_vec
        )
        new_z = jnp.clip(inner, problem.lower_bound, problem.upper_bound)

        ## y ##
        new_y = variables.y + rho_vec * (
            self.alpha * new_z_tilde + (1 - self.alpha) * variables.z - new_z
        )

        return variables.replace(
            x_tilde=new_x_tilde,
            z_tilde=new_z_tilde,
            x=new_x,
            z=new_z,
            y=new_y,
        )
