import jax
import jax.numpy as jnp
from flax import struct

from .optimization_base import (
    ConstraintLinearization,
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
        constraint_lin: ConstraintLinearization,
    ) -> OptimizationVariables:
        # form rho vector
        rho_inequality = self.rho * variables.rho_balance_multiplier
        rho_equality = rho_inequality * self.rho_equality_multiplier
        rho_vec = jnp.where(
            problem.lower_bound == problem.upper_bound,
            rho_equality,
            rho_inequality,
        )

        ## x_tilde ##
        def cg_lhs(v: jax.Array) -> jax.Array:
            # Q v
            a = problem.cost_quad_diag * v
            # sigma I v
            b = self.sigma * v
            # J.T R J v
            c = constraint_lin.h_jvp(v)
            c = rho_vec * c
            c = constraint_lin.h_vjp(c)[0]
            return a + b + c

        # Preconditioner: Use diagonal part (Q + σI) which is always positive
        # definite since σ > 0 and cost_quad_diag >= 0
        precond_diag = problem.cost_quad_diag + self.sigma

        def cg_precond(v: jax.Array) -> jax.Array:
            return v / precond_diag

        zhjx = (
            variables.z
            - constraint_lin.h
            + constraint_lin.h_jvp(constraint_lin.x_lin)
        )
        r_zhjx_y = rho_vec * zhjx - variables.y
        cg_rhs = (
            self.sigma * variables.x
            - problem.cost_linear
            + constraint_lin.h_vjp(r_zhjx_y)[0]
        )

        new_x_tilde, _ = jax.scipy.sparse.linalg.cg(
            A=cg_lhs,
            b=cg_rhs,
            x0=variables.x_tilde,
            M=cg_precond,
        )

        ## z_tilde ##
        new_z_tilde = constraint_lin(new_x_tilde)

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
