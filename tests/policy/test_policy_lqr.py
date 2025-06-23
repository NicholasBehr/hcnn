import jax
import jax.numpy as jnp
import pytest

from hcnn.policy import PolicyLQR
from hcnn.system_dynamics import SystemDynamicsLinear

from .test_policy_base import TestPolicyBase


class TestPolicyLQR(TestPolicyBase):
    """Test class for PolicyLQR."""

    @pytest.fixture
    def sys_dynamics(self, state_dim, action_dim) -> SystemDynamicsLinear:
        key = jax.random.key(0)
        return SystemDynamicsLinear.factory(
            state_matrix=jax.random.normal(key, (state_dim, state_dim)),
            action_matrix=jax.random.normal(key, (state_dim, action_dim)),
        )

    @pytest.fixture
    def policy(self, sys_dynamics: SystemDynamicsLinear):
        return PolicyLQR.factory(
            system_linear=sys_dynamics,
            state_cost=jnp.eye(sys_dynamics.state_dim),
            input_cost=jnp.eye(sys_dynamics.action_dim),
        )

    @pytest.fixture(autouse=True)
    def setup(
        self,
        policy: PolicyLQR,
        sys_dynamics: SystemDynamicsLinear,
    ):
        self.policy = policy
        self.sys_dynamics = sys_dynamics
        self.state_dim = sys_dynamics.state_dim
        self.action_dim = sys_dynamics.action_dim

    def test_instance_creation(self):
        assert isinstance(self.policy, PolicyLQR)
        assert self.policy.state_dim == self.state_dim
        assert self.policy.action_dim == self.action_dim
        assert self.policy.riccati_matrix.shape == (
            self.state_dim,
            self.state_dim,
        )
        assert self.policy.gain_matrix.shape == (
            self.action_dim,
            self.state_dim,
        )

    def test_valid_instance_creation(self):
        assert self.policy.riccati_matrix.shape == (
            self.sys_dynamics.state_dim,
            self.sys_dynamics.state_dim,
        )
        assert self.policy.gain_matrix.shape == (
            self.sys_dynamics.action_dim,
            self.sys_dynamics.state_dim,
        )

    def test_non_symmetric_state_cost_raises(self):
        state_cost = jnp.array([[1.0, 2.0], [0.0, 1.0]])
        input_cost = jnp.eye(self.action_dim)

        with pytest.raises(
            ValueError, match="State cost matrix must be symmetric!"
        ):
            PolicyLQR.factory(self.sys_dynamics, state_cost, input_cost)

    def test_non_psd_state_cost_raises(self):
        state_cost = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        input_cost = jnp.eye(self.action_dim)

        with pytest.raises(ValueError, match="State cost matrix is not PSD!"):
            PolicyLQR.factory(self.sys_dynamics, state_cost, input_cost)

    def test_non_symmetric_input_cost_raises(self):
        state_cost = jnp.eye(self.state_dim)
        input_cost = jnp.array([[1.0, 2.0], [0.0, 1.0]])

        with pytest.raises(
            ValueError, match="Input cost matrix must be symmetric!"
        ):
            PolicyLQR.factory(self.sys_dynamics, state_cost, input_cost)

    def test_non_pd_input_cost_raises(self):
        state_cost = jnp.eye(self.state_dim)
        input_cost = jnp.array([[0.0]])

        with pytest.raises(ValueError, match="Input cost matrix is not PD!"):
            PolicyLQR.factory(self.sys_dynamics, state_cost, input_cost)

    def test_unstabilizable_system_raises(self):
        state_matrix = jnp.array([[1.2, 0.0], [0.0, 1.1]])
        input_matrix = jnp.zeros((2, 1))
        system = SystemDynamicsLinear.factory(
            state_matrix=state_matrix, action_matrix=input_matrix
        )

        state_cost = jnp.eye(2)
        input_cost = jnp.eye(1)

        with pytest.raises(ValueError, match="is not stabilizable"):
            PolicyLQR.factory(system, state_cost, input_cost)

    def test_undetectable_system_raises(self):
        state_matrix = jnp.array([[1.2, 0.0], [0.0, 1.1]])
        input_matrix = jnp.eye(2)
        system = SystemDynamicsLinear.factory(
            state_matrix=state_matrix, action_matrix=input_matrix
        )

        state_cost = jnp.array([[1.0, 0.0], [0.0, 0.0]])
        input_cost = jnp.eye(2)

        with pytest.raises(ValueError, match="is not detectable"):
            PolicyLQR.factory(system, state_cost, input_cost)

    def test_state_cost_shape_mismatch_raises(self):
        state_cost = jnp.eye(self.state_dim + 1)
        input_cost = jnp.eye(self.action_dim)

        with pytest.raises(ValueError, match="State dimension missmatch!"):
            PolicyLQR.factory(self.sys_dynamics, state_cost, input_cost)

    def test_input_cost_shape_mismatch_raises(self):
        state_cost = jnp.eye(self.state_dim)
        input_cost = jnp.eye(self.action_dim + 1)

        with pytest.raises(ValueError, match="Input dimension missmatch!"):
            PolicyLQR.factory(self.sys_dynamics, state_cost, input_cost)

    def test_dare_solver_outputs_valid_riccati_and_gain(self):
        state_matrix = jnp.array([[1.0, 1.0], [0.0, 1.0]])
        input_matrix = jnp.array([[0.0], [1.0]])
        state_cost = jnp.eye(2)
        input_cost = jnp.array([[1.0]])

        system = SystemDynamicsLinear.factory(
            state_matrix=state_matrix, action_matrix=input_matrix
        )
        policy = PolicyLQR.factory(system, state_cost, input_cost)
        riccati_matrix = policy.riccati_matrix
        gain_matrix = policy.gain_matrix

        assert jnp.allclose(riccati_matrix, riccati_matrix.T, atol=1e-6)

        closed_loop = state_matrix - input_matrix @ gain_matrix
        eigvals = jnp.linalg.eigvals(closed_loop)
        assert jnp.all(jnp.abs(eigvals) < 1.0), (
            "Closed-loop system is not stable"
        )

        r_inv_bt_x = jnp.linalg.solve(
            input_cost + input_matrix.T @ riccati_matrix @ input_matrix,
            input_matrix.T @ riccati_matrix,
        )
        dare_lhs = riccati_matrix
        dare_rhs = (
            state_matrix.T @ riccati_matrix @ state_matrix
            - state_matrix.T
            @ riccati_matrix
            @ input_matrix
            @ r_inv_bt_x
            @ state_matrix
            + state_cost
        )
        assert jnp.allclose(dare_lhs, dare_rhs, atol=1e-6), (
            "DARE equation not satisfied"
        )

    def test_dare_solver_reference_case_1(self):
        state_matrix = jnp.array([[-0.9, -0.3], [0.7, 0.1]])
        input_matrix = jnp.array([[1.0], [1.0]])
        state_cost = jnp.array([[1.0, 0.0], [0.0, 3.0]])
        input_cost = jnp.array([[0.1]])

        x_expected = jnp.array([[4.7687, 0.9438], [0.9438, 3.2369]])
        k_expected = jnp.array([[-0.2216, -0.1297]])

        system = SystemDynamicsLinear.factory(
            state_matrix=state_matrix, action_matrix=input_matrix
        )
        policy = PolicyLQR.factory(system, state_cost, input_cost)
        x_computed = policy.riccati_matrix
        k_computed = policy.gain_matrix

        assert jnp.allclose(x_computed, x_expected, atol=1e-3), (
            "Riccati matrix X mismatch"
        )
        assert jnp.allclose(k_computed, k_expected, atol=1e-3), (
            "LQR gain K mismatch"
        )

    def test_dare_solver_reference_case_2(self):
        state_matrix = jnp.array([[1.2, 0.5], [0, 1.1]])
        input_matrix = jnp.array([[0.0], [1.0]])
        state_cost = jnp.eye(2)
        input_cost = jnp.eye(1)

        x_expected = jnp.array([[8.1493, 4.0236], [4.0236, 4.0838]])
        k_expected = jnp.array([[0.949749, 1.2793549]])

        system = SystemDynamicsLinear.factory(
            state_matrix=state_matrix, action_matrix=input_matrix
        )
        policy = PolicyLQR.factory(system, state_cost, input_cost)
        x_computed = policy.riccati_matrix
        k_computed = policy.gain_matrix

        assert jnp.allclose(x_computed, x_expected, atol=1e-3), (
            "Riccati matrix X mismatch"
        )
        assert jnp.allclose(k_computed, k_expected, atol=1e-3), (
            "LQR gain K mismatch"
        )
