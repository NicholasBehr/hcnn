import jax
import jax.numpy as jnp
from flax import nnx

from hcnn.policy.policy import Policy
from hcnn.system.system_linear import SystemLinear


class PolicyLQR(Policy):
    state_cost: jax.Array
    input_cost: jax.Array
    riccati_matrix: jax.Array
    gain_matrix: jax.Array
    test: nnx.Variable

    def __init__(self, system_linear: SystemLinear, state_cost: jax.Array, input_cost: jax.Array):
        tol: float = 1e-8
        state_matrix = system_linear.state_matrix
        input_matrix = system_linear.input_matrix
        state_cost = jnp.atleast_2d(state_cost)
        input_cost = jnp.atleast_2d(input_cost)

        # (1) verify cost matrices
        tol: float = 1e-8
        if not jnp.allclose(state_cost, state_cost.T, atol=tol):
            raise ValueError("State cost matrix must be symmetric!")
        if not jnp.allclose(input_cost, input_cost.T, atol=tol):
            raise ValueError("Input cost matrix must be symmetric!")

        matrix_q_eigenvals = jnp.linalg.eigvals(state_cost)
        matrix_r_eigenvals = jnp.linalg.eigvals(input_cost)

        if not jnp.all(matrix_q_eigenvals >= -tol):
            raise ValueError("State cost matrix is not PSD!")
        if not jnp.all(matrix_r_eigenvals >= +tol):
            raise ValueError("Input cost matrix is not PD!")

        # (2) verify existence of solution
        # 1. R is PD (done)
        # 2. (A, B) is stabilizable
        if not self.is_stabilizable(state_matrix, input_matrix):
            raise ValueError("(A, B) is not stabilizable!")
        # 3. (A, Q^{1/2}) is detectable
        u, s, vt = jnp.linalg.svd(state_cost, hermitian=True)
        matrix_q_sqrt = u @ jnp.diag(jnp.power(s, 0.5)) @ vt
        if not self.is_stabilizable(state_matrix.T, matrix_q_sqrt.T):
            raise ValueError("(A, Q^{1/2}) is not detectable!")

        # (3) solve DARE using the system matrices
        riccati_matrix, gain_matrix = self.solve_discreate_are(
            state_matrix,
            input_matrix,
            state_cost,
            input_cost,
        )

        # (4) create instance
        self.state_cost = state_cost
        self.input_cost = input_cost
        self.riccati_matrix = riccati_matrix
        self.gain_matrix = gain_matrix
        self.test = nnx.Param(jnp.array(1.0))

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
    def solve_discreate_are(
        state_matrix: jax.Array,
        input_matrix: jax.Array,
        state_cost: jax.Array,
        input_cost: jax.Array,
        tol: float = 1e-8,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Solve the discrete algebraic Riccati equation (DARE) using policy iteration.
        The DARE is given by:
        X = A^T * X * A - (A^T * X * B) * (R + B^T * X * B)^(-1) * (B^T * X * A) + Q
        where:
        - A is the state transition matrix
        - B is the input matrix
        - Q is the state cost matrix
        - R is the input cost matrix
        - X is the solution to the DARE
        """

        @jax.jit
        def policy_improvement(riccati_matrix):
            return jnp.linalg.solve(
                input_cost + input_matrix.T @ riccati_matrix @ input_matrix,
                input_matrix.T @ riccati_matrix @ state_matrix,
            )

        @jax.jit
        def policy_evaluation(riccati_matrix, gain_matrix):
            return (
                state_cost
                + state_matrix.T @ riccati_matrix @ state_matrix
                - state_matrix.T @ riccati_matrix @ input_matrix @ gain_matrix
            )

        gain_matrix = jnp.zeros((input_matrix.shape[1], state_matrix.shape[0]))
        riccati_matrix = state_cost
        for _ in range(500):
            gain_matrix = policy_improvement(riccati_matrix)
            new_riccati_matrix = policy_evaluation(riccati_matrix, gain_matrix)
            if jnp.linalg.norm(new_riccati_matrix - riccati_matrix) < tol:
                break
            riccati_matrix = new_riccati_matrix
        return riccati_matrix, gain_matrix

    def __call__(self, state: jax.Array) -> jax.Array:
        return -self.gain_matrix @ state
