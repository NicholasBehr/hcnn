import jax
import jax.numpy as jnp
import pytest

from hcnn.policy.policy_lqr import PolicyLQR
from hcnn.system_dynamics.system_dynamics_linear import SystemDynamicsLinear
from tests.policy.policy_base import *


@pytest.fixture
def dynamics_linear():
    key = jax.random.key(0)
    return SystemDynamicsLinear.factory(
        state_matrix=jax.random.normal(key, (2, 2)),
        input_matrix=jax.random.normal(key, (2, 1)),
    )


@pytest.fixture
def instance(dynamics_linear: SystemDynamicsLinear):
    return PolicyLQR.factory(
        system_linear=dynamics_linear,
        state_cost=jnp.eye(dynamics_linear.state_dim),
        input_cost=jnp.eye(dynamics_linear.action_dim),
    )


def test_valid_instance_creation(instance: PolicyLQR, dynamics_linear: SystemDynamicsLinear):
    assert instance.riccati_matrix.shape == (dynamics_linear.state_dim, dynamics_linear.state_dim)
    assert instance.gain_matrix.shape == (dynamics_linear.action_dim, dynamics_linear.state_dim)


def test_non_symmetric_state_cost_raises(dynamics_linear: SystemDynamicsLinear):
    state_cost = jnp.array([[1.0, 2.0], [0.0, 1.0]])
    input_cost = jnp.eye(1)

    with pytest.raises(ValueError, match="State cost matrix must be symmetric!"):
        PolicyLQR.factory(dynamics_linear, state_cost, input_cost)


def test_non_psd_state_cost_raises(dynamics_linear: SystemDynamicsLinear):
    state_cost = jnp.array([[1.0, 0.0], [0.0, -1.0]])
    input_cost = jnp.eye(1)

    with pytest.raises(ValueError, match="State cost matrix is not PSD!"):
        PolicyLQR.factory(dynamics_linear, state_cost, input_cost)


def test_non_symmetric_input_cost_raises(dynamics_linear: SystemDynamicsLinear):
    state_cost = jnp.eye(2)
    input_cost = jnp.array([[1.0, 2.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="Input cost matrix must be symmetric!"):
        PolicyLQR.factory(dynamics_linear, state_cost, input_cost)


def test_non_pd_input_cost_raises(dynamics_linear: SystemDynamicsLinear):
    state_cost = jnp.eye(2)
    input_cost = jnp.array([[0.0]])

    with pytest.raises(ValueError, match="Input cost matrix is not PD!"):
        PolicyLQR.factory(dynamics_linear, state_cost, input_cost)


def test_unstabilizable_system_raises():
    state_matrix = jnp.array([[1.2, 0.0], [0.0, 1.1]])
    input_matrix = jnp.zeros((2, 1))
    system = SystemDynamicsLinear.factory(state_matrix=state_matrix, input_matrix=input_matrix)

    state_cost = jnp.eye(2)
    input_cost = jnp.eye(1)

    with pytest.raises(ValueError, match="is not stabilizable"):
        PolicyLQR.factory(system, state_cost, input_cost)


def test_undetectable_system_raises():
    state_matrix = jnp.array([[1.2, 0.0], [0.0, 1.1]])
    input_matrix = jnp.eye(2)  # stabilizable, but Q will cause detectability to fail
    system = SystemDynamicsLinear.factory(state_matrix=state_matrix, input_matrix=input_matrix)

    state_cost = jnp.array([[1.0, 0.0], [0.0, 0.0]])  # makes Q sqrt only affect one state
    input_cost = jnp.eye(2)

    with pytest.raises(ValueError, match="is not detectable"):
        PolicyLQR.factory(system, state_cost, input_cost)


def test_state_cost_shape_mismatch_raises(dynamics_linear: SystemDynamicsLinear):
    state_cost = jnp.eye(3)  # should be 2x2
    input_cost = jnp.eye(1)

    with pytest.raises(ValueError, match="State dimension missmatch!"):
        PolicyLQR.factory(dynamics_linear, state_cost, input_cost)


def test_input_cost_shape_mismatch_raises(dynamics_linear: SystemDynamicsLinear):
    state_cost = jnp.eye(2)
    input_cost = jnp.eye(2)  # should be 1x1

    with pytest.raises(ValueError, match="Input dimension missmatch!"):
        PolicyLQR.factory(dynamics_linear, state_cost, input_cost)


def test_dare_solver_outputs_valid_riccati_and_gain():
    # Simple stable system
    state_matrix = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    input_matrix = jnp.array([[0.0], [1.0]])
    state_cost = jnp.eye(2)
    input_cost = jnp.array([[1.0]])

    system = SystemDynamicsLinear.factory(state_matrix=state_matrix, input_matrix=input_matrix)
    policy = PolicyLQR.factory(system, state_cost, input_cost)
    riccati_matrix = policy.riccati_matrix
    gain_matrix = policy.gain_matrix

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


def test_dare_solver_reference_case_1():
    # Matrices from the MATLAB example
    state_matrix = jnp.array([[-0.9, -0.3], [0.7, 0.1]])
    input_matrix = jnp.array([[1.0], [1.0]])
    state_cost = jnp.array([[1.0, 0.0], [0.0, 3.0]])
    input_cost = jnp.array([[0.1]])

    # Expected results (from MATLAB idare)
    x_expected = jnp.array([[4.7687, 0.9438], [0.9438, 3.2369]])
    k_expected = jnp.array([[-0.2216, -0.1297]])

    system = SystemDynamicsLinear.factory(state_matrix=state_matrix, input_matrix=input_matrix)
    policy = PolicyLQR.factory(system, state_cost, input_cost)
    x_computed = policy.riccati_matrix
    k_computed = policy.gain_matrix

    assert jnp.allclose(x_computed, x_expected, atol=1e-3), "Riccati matrix X mismatch"
    assert jnp.allclose(k_computed, k_expected, atol=1e-3), "LQR gain K mismatch"


def test_dare_solver_reference_case_2():
    # Matrices from the MATLAB example
    state_matrix = jnp.array([[1.2, 0.5], [0, 1.1]])
    input_matrix = jnp.array([[0.0], [1.0]])
    state_cost = jnp.eye(2)
    input_cost = jnp.eye(1)

    # Expected results (from MATLAB idare)
    x_expected = jnp.array([[8.1493, 4.0236], [4.0236, 4.0838]])
    k_expected = jnp.array([[0.949749, 1.2793549]])

    system = SystemDynamicsLinear.factory(state_matrix=state_matrix, input_matrix=input_matrix)
    policy = PolicyLQR.factory(system, state_cost, input_cost)
    x_computed = policy.riccati_matrix
    k_computed = policy.gain_matrix

    assert jnp.allclose(x_computed, x_expected, atol=1e-3), "Riccati matrix X mismatch"
    assert jnp.allclose(k_computed, k_expected, atol=1e-3), "LQR gain K mismatch"
