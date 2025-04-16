import jax.numpy as jnp
import pytest

from hcnn.policy.policy_lqr import PolicyLQR
from hcnn.system.system_linear import SystemLinear


@pytest.fixture
def stable_system():
    state_matrix = jnp.array([[0.9, 0.0], [0.0, 0.95]])
    input_matrix = jnp.array([[0.1], [0.05]])
    return SystemLinear(state_matrix=state_matrix, input_matrix=input_matrix)


def test_factory_valid(stable_system):
    state_cost = jnp.eye(2)
    input_cost = jnp.eye(1)

    policy = PolicyLQR.factory(stable_system, state_cost, input_cost)

    assert isinstance(policy, PolicyLQR)
    assert jnp.allclose(policy.state_cost, state_cost)
    assert jnp.allclose(policy.input_cost, input_cost)
    assert policy.riccati_matrix.shape == state_cost.shape
    assert policy.gain_matrix.shape == (input_cost.shape[0], state_cost.shape[0])


def test_non_symmetric_state_cost_raises(stable_system):
    state_cost = jnp.array([[1.0, 2.0], [0.0, 1.0]])
    input_cost = jnp.eye(1)

    with pytest.raises(ValueError, match="State cost matrix must be symmetric!"):
        PolicyLQR.factory(stable_system, state_cost, input_cost)


def test_non_psd_state_cost_raises(stable_system):
    state_cost = jnp.array([[1.0, 0.0], [0.0, -1.0]])
    input_cost = jnp.eye(1)

    with pytest.raises(ValueError, match="State cost matrix is not PSD!"):
        PolicyLQR.factory(stable_system, state_cost, input_cost)


def test_non_symmetric_input_cost_raises(stable_system):
    state_cost = jnp.eye(2)
    input_cost = jnp.array([[1.0, 2.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="Input cost matrix must be symmetric!"):
        PolicyLQR.factory(stable_system, state_cost, input_cost)


def test_non_pd_input_cost_raises(stable_system):
    state_cost = jnp.eye(2)
    input_cost = jnp.array([[0.0]])

    with pytest.raises(ValueError, match="Input cost matrix is not PD!"):
        PolicyLQR.factory(stable_system, state_cost, input_cost)


def test_unstabilizable_system_raises():
    state_matrix = jnp.array([[1.2, 0.0], [0.0, 1.1]])
    input_matrix = jnp.zeros((2, 1))
    system = SystemLinear(state_matrix=state_matrix, input_matrix=input_matrix)

    state_cost = jnp.eye(2)
    input_cost = jnp.eye(1)

    with pytest.raises(ValueError, match="is not stabilizable"):
        PolicyLQR.factory(system, state_cost, input_cost)


def test_undetectable_system_raises():
    state_matrix = jnp.array([[1.2, 0.0], [0.0, 1.1]])
    input_matrix = jnp.eye(2)  # stabilizable, but Q will cause detectability to fail
    system = SystemLinear(state_matrix=state_matrix, input_matrix=input_matrix)

    state_cost = jnp.array([[1.0, 0.0], [0.0, 0.0]])  # makes Q sqrt only affect one state
    input_cost = jnp.eye(2)

    with pytest.raises(ValueError, match="is not detectable"):
        PolicyLQR.factory(system, state_cost, input_cost)


def test_direct_instantiation_raises():
    state_cost = jnp.zeros((2, 2))
    input_cost = jnp.zeros((2, 2))
    riccati_matrix = jnp.zeros((2, 2))
    gain_matrix = jnp.zeros((2, 2))
    with pytest.raises(TypeError, match="Direct instantiation is not allowed"):
        PolicyLQR(state_cost, input_cost, riccati_matrix, gain_matrix)


def test_state_cost_shape_mismatch_raises(stable_system):
    state_cost = jnp.eye(3)  # should be 2x2
    input_cost = jnp.eye(1)

    with pytest.raises(ValueError, match="State dimension missmatch!"):
        PolicyLQR.factory(stable_system, state_cost, input_cost)


def test_input_cost_shape_mismatch_raises(stable_system):
    state_cost = jnp.eye(2)
    input_cost = jnp.eye(2)  # should be 1x1

    with pytest.raises(ValueError, match="Input dimension missmatch!"):
        PolicyLQR.factory(stable_system, state_cost, input_cost)


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
