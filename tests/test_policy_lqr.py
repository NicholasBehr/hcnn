import jax.numpy as jnp

from hcnn.policy.policy_lqr import PolicyLQR


# -- dare_solver --
def test_dare_solver_outputs_valid_riccati_and_gain():
    # Simple stable system
    state_matrix = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    input_matrix = jnp.array([[0.0], [1.0]])
    state_cost = jnp.eye(2)
    input_cost = jnp.array([[1.0]])

    riccati_matrix, gain_matrix = PolicyLQR.dare_solver(
        state_matrix, input_matrix, state_cost, input_cost
    )

    # 1. Riccati matrix should be symmetric
    assert jnp.allclose(riccati_matrix, riccati_matrix.T, atol=1e-6)

    # 2. LQR gain should stabilize the closed-loop system
    closed_loop = state_matrix - input_matrix @ gain_matrix
    eigvals = jnp.linalg.eigvals(closed_loop)
    assert jnp.all(jnp.abs(eigvals) < 1.0), "Closed-loop system is not stable"

    # 3. Riccati matrix should approximately satisfy the DARE
    r_inv_bt_x = jnp.linalg.solve(
        input_cost + input_matrix.T @ riccati_matrix @ input_matrix, input_matrix.T @ riccati_matrix
    )
    dare_lhs = riccati_matrix
    dare_rhs = (
        state_matrix.T @ riccati_matrix @ state_matrix
        - state_matrix.T @ riccati_matrix @ input_matrix @ r_inv_bt_x @ state_matrix
        + state_cost
    )
    assert jnp.allclose(dare_lhs, dare_rhs, atol=1e-6), "DARE equation not satisfied"


def test_dare_solver_reference_case():
    # Matrices from the MATLAB example
    state_matrix = jnp.array([[-0.9, -0.3], [0.7, 0.1]])
    input_matrix = jnp.array([[1.0], [1.0]])
    state_cost = jnp.array([[1.0, 0.0], [0.0, 3.0]])
    input_cost = jnp.array([[0.1]])

    # Expected results (from MATLAB idare)
    x_expected = jnp.array([[4.7687, 0.9438], [0.9438, 3.2369]])
    k_expected = jnp.array([[-0.2216, -0.1297]])

    x_computed, k_computed = PolicyLQR.dare_solver(
        state_matrix, input_matrix, state_cost, input_cost
    )

    assert jnp.allclose(x_computed, x_expected, atol=1e-3), "Riccati matrix X mismatch"
    assert jnp.allclose(k_computed, k_expected, atol=1e-3), "LQR gain K mismatch"
