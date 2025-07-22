import jax.numpy as jnp
import pytest

from hcnn.system_dynamics.system_dynamics_continuous import Integrator
from hcnn.system_dynamics.system_dynamics_double_pendulum import (
    SystemDynamicsDoublePendulum,
)

from .test_system_dynamics_base import TestSystemDynamicsBase


class TestSystemDynamicsDoublePendulum(TestSystemDynamicsBase):
    """Test class for SystemDynamicsDoublePendulum."""

    state_dim: int = 4  # [θ₁, θ₂, θ̇₁, θ̇₂]
    action_dim: int = 1  # [τ₁]

    @pytest.fixture
    def sys_dynamics(self) -> SystemDynamicsDoublePendulum:
        return SystemDynamicsDoublePendulum.factory(
            m1=1.0,
            m2=1.0,
            l1=1.0,
            l2=1.0,
            g=9.81,
            b1=0.1,
            b2=0.1,
            dt=0.01,
            integrator=Integrator.Dopri5,
        )

    def test_instance_creation_double_pendulum(
        self, sys_dynamics: SystemDynamicsDoublePendulum
    ):
        """Test that the factory method creates a valid instance."""
        assert isinstance(sys_dynamics, SystemDynamicsDoublePendulum)
        assert sys_dynamics.state_dim == self.state_dim
        assert sys_dynamics.action_dim == self.action_dim
        assert sys_dynamics.m1 == 1.0
        assert sys_dynamics.m2 == 1.0
        assert sys_dynamics.l1 == 1.0
        assert sys_dynamics.l2 == 1.0
        assert sys_dynamics.g == 9.81
        assert sys_dynamics.b1 == 0.1
        assert sys_dynamics.b2 == 0.1
        assert sys_dynamics.dt == 0.01
        assert isinstance(sys_dynamics.integrator, Integrator)

    def test_factory_validation(self):
        """Test that factory method validates parameters correctly."""
        # Test negative masses
        with pytest.raises(AssertionError, match="Masses must be positive"):
            SystemDynamicsDoublePendulum.factory(m1=-1.0)

        with pytest.raises(AssertionError, match="Masses must be positive"):
            SystemDynamicsDoublePendulum.factory(m2=-1.0)

        # Test negative lengths
        with pytest.raises(AssertionError, match="Lengths must be positive"):
            SystemDynamicsDoublePendulum.factory(l1=-1.0)

        with pytest.raises(AssertionError, match="Lengths must be positive"):
            SystemDynamicsDoublePendulum.factory(l2=-1.0)

        # Test negative gravity
        with pytest.raises(
            AssertionError, match="Gravitational acceleration must be positive"
        ):
            SystemDynamicsDoublePendulum.factory(g=-9.81)

        # Test negative damping
        with pytest.raises(
            AssertionError, match="Damping coefficients must be non-negative"
        ):
            SystemDynamicsDoublePendulum.factory(b1=-0.1)

        with pytest.raises(
            AssertionError, match="Damping coefficients must be non-negative"
        ):
            SystemDynamicsDoublePendulum.factory(b2=-0.1)

        # Test negative timestep
        with pytest.raises(AssertionError, match="Timestep must be positive"):
            SystemDynamicsDoublePendulum.factory(dt=-0.01)

    def test_continuous_dynamics_method(
        self, sys_dynamics: SystemDynamicsDoublePendulum
    ):
        """Test the continuous_dynamics method directly."""
        # Test equilibrium (hanging down)
        state = jnp.array([0.0, 0.0, 0.0, 0.0])  # Both links hanging down
        action = jnp.array([0.0])  # No torque

        derivative = sys_dynamics._continuous_dynamics(state, action)
        assert derivative.shape == (4,)

        # At equilibrium with no torque, angular velocities should be zero
        # and angular accelerations should be zero (ignoring numerical errors)
        assert jnp.abs(derivative[0]) < 1e-6  # θ̇₁ = 0
        assert jnp.abs(derivative[1]) < 1e-6  # θ̇₂ = 0
        assert jnp.abs(derivative[2]) < 1e-6  # θ̈₁ = 0
        assert jnp.abs(derivative[3]) < 1e-6  # θ̈₂ = 0

    def test_continuous_dynamics_with_torque(
        self, sys_dynamics: SystemDynamicsDoublePendulum
    ):
        """Test that applied torque produces expected accelerations."""
        # Start from equilibrium with positive torque
        state = jnp.array([0.0, 0.0, 0.0, 0.0])
        action = jnp.array([1.0])  # Positive torque at first joint

        derivative = sys_dynamics._continuous_dynamics(state, action)

        # With positive torque, first link should have positive acceleration
        assert derivative[2] > 0  # θ̈₁ > 0

    def test_gravity_effect(self, sys_dynamics: SystemDynamicsDoublePendulum):
        """Test that gravity produces expected restoring forces."""
        # Test with first link at 90 degrees (horizontal)
        state = jnp.array([jnp.pi / 2, 0.0, 0.0, 0.0])
        action = jnp.array([0.0])

        derivative = sys_dynamics._continuous_dynamics(state, action)

        # Gravity should produce negative angular acceleration
        # (restoring toward equilibrium)
        assert derivative[2] < 0  # θ̈₁ < 0 (gravity pulls it back)

    def test_energy_conservation_no_damping(self):
        """Test energy conservation for undamped system."""
        # Create system without damping
        sys_dynamics = SystemDynamicsDoublePendulum.factory(
            m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81, b1=0.0, b2=0.0, dt=0.001
        )

        # Start with some initial conditions
        state = jnp.ones(4)
        action = jnp.array([0.0])  # No external torque

        initial_energy = sys_dynamics.get_energy(state)

        # Simulate for a few steps
        current_state = state
        for _ in range(50):
            current_state = sys_dynamics(current_state, action)

        final_energy = sys_dynamics.get_energy(current_state)

        # Energy should be approximately conserved (within integration error)
        assert jnp.abs(final_energy - initial_energy) < 0.01

    def test_energy_dissipation_with_damping(
        self, sys_dynamics: SystemDynamicsDoublePendulum
    ):
        """Test that damping dissipates energy."""
        # Start with some initial velocity
        state = jnp.array([0.1, 0.2, 1.0, 0.5])  # Small angles, some velocities
        action = jnp.array([0.0])  # No external torque

        initial_energy = sys_dynamics.get_energy(state)

        # Simulate for several steps
        current_state = state
        for _ in range(50):
            current_state = sys_dynamics(current_state, action)

        final_energy = sys_dynamics.get_energy(current_state)

        # Energy should decrease due to damping
        assert final_energy < initial_energy

    def test_different_integrators(self):
        """Test that different integrators work correctly."""
        integrators = [Integrator.Euler, Integrator.Tsit5, Integrator.Dopri5]

        state = jnp.array([0.1, 0.1, 0.0, 0.0])
        action = jnp.array([0.0])

        results = []
        for integrator in integrators:
            sys_dynamics = SystemDynamicsDoublePendulum.factory(
                dt=0.01, integrator=integrator
            )
            next_state = sys_dynamics(state, action)
            results.append(next_state)

        # All integrators should produce finite results
        for result in results:
            assert jnp.all(jnp.isfinite(result))

        # Results should be similar but not identical
        for i in range(1, len(results)):
            # Check that results are close but not exactly the same
            assert not jnp.allclose(results[0], results[i], atol=1e-10)
            assert jnp.allclose(results[0], results[i], rtol=0.1)

    def test_both_links_hanging_down(self, sys_dynamics):
        """Test Case 1: Both links hanging down (θ₁=0, θ₂=0)"""
        state = jnp.array([0.0, 0.0, 0.0, 0.0])
        positions = sys_dynamics.forward_kinematics(state)
        x1, y1, x2, y2 = positions

        # Link 1 end: x1 = l₁sin(0) = 0, y1 = -l₁cos(0) = -l₁
        expected_x1, expected_y1 = 0.0, -sys_dynamics.l1

        # Link 2 end: x2 = 0, y2 = -l₁ - l₂
        expected_x2 = 0.0
        expected_y2 = -(sys_dynamics.l1 + sys_dynamics.l2)

        assert jnp.abs(x1 - expected_x1) < 1e-6
        assert jnp.abs(y1 - expected_y1) < 1e-6
        assert jnp.abs(x2 - expected_x2) < 1e-6
        assert jnp.abs(y2 - expected_y2) < 1e-6

    def test_first_link_horizontal_second_straight(self, sys_dynamics):
        """Test Case 2: First link horizontal right (θ₁=π/2, θ₂=0)"""
        state = jnp.array([jnp.pi / 2, 0.0, 0.0, 0.0])
        positions = sys_dynamics.forward_kinematics(state)
        x1, y1, x2, y2 = positions

        # Link 1 end: x1 = l₁sin(π/2) = l₁, y1 = -l₁cos(π/2) = 0
        expected_x1, expected_y1 = sys_dynamics.l1, 0.0

        # Link 2 end: x2 = l₁ + l₂, y2 = 0
        expected_x2 = sys_dynamics.l1 + sys_dynamics.l2
        expected_y2 = 0.0

        assert jnp.abs(x1 - expected_x1) < 1e-6
        assert jnp.abs(y1 - expected_y1) < 1e-6
        assert jnp.abs(x2 - expected_x2) < 1e-6
        assert jnp.abs(y2 - expected_y2) < 1e-6

    def test_first_horizontal_second_perpendicular(self, sys_dynamics):
        """Test Case 3: First horizontal, second perpendicular."""
        state = jnp.array([jnp.pi / 2, jnp.pi / 2, 0.0, 0.0])
        positions = sys_dynamics.forward_kinematics(state)
        x1, y1, x2, y2 = positions

        # Link 1 end: x1 = l₁, y1 = 0
        expected_x1, expected_y1 = sys_dynamics.l1, 0.0

        # Link 2 end: x2 = l₁ + l₂sin(π) = l₁, y2 = 0 + l₂
        expected_x2 = sys_dynamics.l1
        expected_y2 = sys_dynamics.l2

        assert jnp.abs(x1 - expected_x1) < 1e-6
        assert jnp.abs(y1 - expected_y1) < 1e-6
        assert jnp.abs(x2 - expected_x2) < 1e-6
        assert jnp.abs(y2 - expected_y2) < 1e-6

    def test_first_horizontal_second_opposite(self, sys_dynamics):
        """Test Case 4: First horizontal (θ₁=π/2), second opposite (θ₂=π)"""
        state = jnp.array([jnp.pi / 2, jnp.pi, 0.0, 0.0])
        positions = sys_dynamics.forward_kinematics(state)
        x1, y1, x2, y2 = positions

        # Link 1 end: x1 = l₁, y1 = 0
        expected_x1, expected_y1 = sys_dynamics.l1, 0.0

        # Link 2 end: x2 = l₁ + l₂sin(3π/2) = l₁ - l₂, y2 = 0
        expected_x2 = sys_dynamics.l1 - sys_dynamics.l2
        expected_y2 = 0.0

        assert jnp.abs(x1 - expected_x1) < 1e-6
        assert jnp.abs(y1 - expected_y1) < 1e-6
        assert jnp.abs(x2 - expected_x2) < 1e-6
        assert jnp.abs(y2 - expected_y2) < 1e-6

    def test_first_down_second_perpendicular(self, sys_dynamics):
        """Test Case 5: First down (θ₁=0), second perpendicular (θ₂=π/2)"""
        state = jnp.array([0.0, jnp.pi / 2, 0.0, 0.0])
        positions = sys_dynamics.forward_kinematics(state)
        x1, y1, x2, y2 = positions

        # Link 1 end: x1 = 0, y1 = -l₁
        expected_x1, expected_y1 = 0.0, -sys_dynamics.l1

        # Link 2 end: x2 = 0 + l₂ = l₂, y2 = -l₁ - 0 = -l₁
        expected_x2 = sys_dynamics.l2
        expected_y2 = -sys_dynamics.l1

        assert jnp.abs(x1 - expected_x1) < 1e-6
        assert jnp.abs(y1 - expected_y1) < 1e-6
        assert jnp.abs(x2 - expected_x2) < 1e-6
        assert jnp.abs(y2 - expected_y2) < 1e-6
