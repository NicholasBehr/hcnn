from dataclasses import dataclass

import jax
import jax.numpy as jnp

from hcnn.policy.policy import Policy
from hcnn.system.system_linear import SystemLinear


@dataclass(frozen=True)
class PolicyLQR(Policy):
    state_cost: jax.Array
    input_cost: jax.Array
    riccati_matrix: jax.Array
    gain_matrix: jax.Array

    def __new__(cls, *args, **kwargs):
        raise TypeError("Direct instantiation is not allowed. Use `PolicyLQR.factory()` instead.")

    @classmethod
    def factory(
        cls, system_linear: SystemLinear, state_cost: jax.Array, input_cost: jax.Array
    ) -> "PolicyLQR":
        tol: float = 1e-8

        # (1) verify cost matrices
        tol: float = 1e-8
        if state_cost.ndim != 2 or input_cost.ndim != 2:  # noqa: PLR2004
            raise ValueError("Cost matrix must be two dimensional!")
        if not jnp.allclose(state_cost, state_cost.T, atol=tol):
            raise ValueError("State cost matrix must be symmetric!")
        if not jnp.allclose(input_cost, input_cost.T, atol=tol):
            raise ValueError("Input cost matrix must be symmetric!")
        if state_cost.shape[0] != system_linear.state_dim:
            raise ValueError("State dimension missmatch!")
        if input_cost.shape[0] != system_linear.input_dim:
            raise ValueError("Input dimension missmatch!")

        matrix_q_eigenvals = jnp.linalg.eigvals(state_cost)
        matrix_r_eigenvals = jnp.linalg.eigvals(input_cost)

        if not jnp.all(matrix_q_eigenvals >= -tol):
            raise ValueError("State cost matrix is not PSD!")
        if not jnp.all(matrix_r_eigenvals >= +tol):
            raise ValueError("Input cost matrix is not PD!")

        # (2) verify existence of solution
        # 1. R is PD (done)
        # 2. (A, B) is stabilizable
        if not cls.is_stabilizable(system_linear.state_matrix, system_linear.input_matrix):
            raise ValueError("(A, B) is not stabilizable!")
        # 3. (A, Q^{1/2}) is detectable
        u, s, vt = jnp.linalg.svd(state_cost, hermitian=True)
        matrix_q_sqrt = u @ jnp.diag(jnp.power(s, 0.5)) @ vt
        if not cls.is_stabilizable(system_linear.state_matrix.T, matrix_q_sqrt.T):
            raise ValueError("(A, Q^{1/2}) is not detectable!")

        # (3) solve DARE using the system matrices
        riccati_matrix, gain_matrix = cls.dare_solver(
            system_linear.state_matrix, system_linear.input_matrix, state_cost, input_cost
        )

        # (4) create instance
        # Bypass __new__ using object.__new__
        obj = object.__new__(cls)
        object.__setattr__(obj, "state_cost", state_cost)
        object.__setattr__(obj, "input_cost", input_cost)
        object.__setattr__(obj, "riccati_matrix", riccati_matrix)
        object.__setattr__(obj, "gain_matrix", gain_matrix)
        return obj

    @staticmethod
    def is_stabilizable(
        state_matrix: jax.Array, input_matrix: jax.Array, tol: float = 1e-7
    ) -> bool:
        n = state_matrix.shape[0]

        for eigval in jnp.linalg.eigvals(state_matrix):
            # Only check modes with |λ| >= 1 (unstable)
            if jnp.abs(eigval) >= 1.0 - tol:
                # Check if the controllability matrix [λI - A, B] has full rank
                test_matrix = jnp.hstack((eigval * jnp.eye(n) - state_matrix, input_matrix))
                if jnp.linalg.matrix_rank(test_matrix, rtol=tol) < n:
                    return False
        return True

    @staticmethod
    def dare_solver(
        state_matrix: jax.Array,
        input_matrix: jax.Array,
        state_cost: jax.Array,
        input_cost: jax.Array,
        tol: float = 1e-8,
    ):
        max_iter = 500

        @jax.jit
        def riccati_iteration(matrix_x):
            input_t_x_input = input_matrix.T @ matrix_x @ input_matrix
            gain_term = state_matrix.T @ matrix_x @ input_matrix
            mid_term = input_cost + input_t_x_input
            mid_solve = jnp.linalg.solve(mid_term, gain_term.T)
            return state_matrix.T @ matrix_x @ state_matrix - gain_term @ mid_solve + state_cost

        riccati_matrix = state_cost
        for _ in range(max_iter):
            riccati_matrix_next = riccati_iteration(riccati_matrix)
            if jnp.linalg.norm(riccati_matrix_next - riccati_matrix) < tol:
                riccati_matrix = riccati_matrix_next
                break
            riccati_matrix = riccati_matrix_next

        # Compute LQR gain: K = (R + Bᵀ X B)⁻¹ Bᵀ X A
        gain_matrix = jnp.linalg.solve(
            input_cost + input_matrix.T @ riccati_matrix @ input_matrix,
            input_matrix.T @ riccati_matrix @ state_matrix,
        )

        return riccati_matrix, gain_matrix

    def get_action(self, state: jax.Array) -> jax.Array:
        return -self.gain_matrix @ state
