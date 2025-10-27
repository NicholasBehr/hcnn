"""
Tests for the SystemDynamicsBike class.
"""

import jax.numpy as jnp
import pytest

from hcnn.system_dynamics.system_dynamics_bike import SystemDynamicsBike

from .test_system_dynamics_base import TestSystemDynamicsBase


class TestSystemDynamicsBike(TestSystemDynamicsBase):
    """Test cases for the kinematic bike model, extending base tests."""

    state_dim: int = 4  # [x, y, θ, v]
    action_dim: int = 2  # [a, δ]

    @pytest.fixture
    def sys_dynamics(self) -> SystemDynamicsBike:
        """Create a bike model instance for testing."""
        return SystemDynamicsBike.factory(wheelbase=2.5, dt=0.1)

    def test_factory_creation(self, sys_dynamics: SystemDynamicsBike):
        """Test that factory method creates valid bike model."""
        assert sys_dynamics.state_dim == 4
        assert sys_dynamics.action_dim == 2
        assert sys_dynamics.wheelbase == 2.5
        assert sys_dynamics.dt == 0.1

    def test_factory_validation(self):
        """Test that factory validates parameters."""
        with pytest.raises(AssertionError):
            SystemDynamicsBike.factory(wheelbase=-1.0)
        with pytest.raises(AssertionError):
            SystemDynamicsBike.factory(dt=-1.0)

    def test_straight_line_motion(self, sys_dynamics: SystemDynamicsBike):
        """Test straight line motion with no steering."""
        # Initial state: [x=0, y=0, theta=0, v=10] (facing east, 10 m/s)
        state = jnp.array([0.0, 0.0, 0.0, 10.0])
        # Action: [a=0, delta=0] (no acceleration, no steering)
        action = jnp.array([0.0, 0.0])

        # Compute state derivative
        state_dot = sys_dynamics._continuous_dynamics(state, action)

        # Expected: [x_dot=10, y_dot=0, theta_dot=0, v_dot=0]
        expected = jnp.array([10.0, 0.0, 0.0, 0.0])

        assert jnp.allclose(state_dot, expected, atol=1e-6)

    def test_turning_motion(self, sys_dynamics: SystemDynamicsBike):
        """Test turning motion with steering input."""
        # Initial state: [x=0, y=0, theta=0, v=10] (facing east, 10 m/s)
        state = jnp.array([0.0, 0.0, 0.0, 10.0])
        # Action: [a=0, delta=0.1] (no acceleration, small left turn)
        action = jnp.array([0.0, 0.1])

        # Compute state derivative
        state_dot = sys_dynamics._continuous_dynamics(state, action)

        # Check that vehicle turns (theta_dot > 0 for left turn)
        assert state_dot[2] > 0  # Positive angular velocity (left turn)
        assert jnp.isclose(state_dot[0], 10.0, atol=1e-6)  # Still moving east
        assert jnp.isclose(state_dot[1], 0.0, atol=1e-6)  # No northward motion
        assert jnp.isclose(state_dot[3], 0.0, atol=1e-6)  # No acceleration

    def test_acceleration(self, sys_dynamics: SystemDynamicsBike):
        """Test acceleration behavior."""
        # Initial state: [x=0, y=0, theta=0, v=5] (facing east, 5 m/s)
        state = jnp.array([0.0, 0.0, 0.0, 5.0])
        # Action: [a=2, delta=0] (2 m/s² acceleration, no steering)
        action = jnp.array([2.0, 0.0])

        # Compute state derivative
        state_dot = sys_dynamics._continuous_dynamics(state, action)

        # Expected: [x_dot=5, y_dot=0, theta_dot=0, v_dot=2]
        expected = jnp.array([5.0, 0.0, 0.0, 2.0])

        assert jnp.allclose(state_dot, expected, atol=1e-6)

    def test_negative_velocity(self, sys_dynamics: SystemDynamicsBike):
        """Test behavior with negative (reverse) velocity."""
        # State with negative velocity (reversing)
        state = jnp.array([0.0, 0.0, jnp.pi, -5.0])  # Facing west, -5 m/s
        action = jnp.array([0.0, 0.1])  # Small steering

        state_dot = sys_dynamics._continuous_dynamics(state, action)

        # Should move in negative x direction (backward)
        assert state_dot[0] > 0  # Actually moving east due to negative v and pi
        assert state_dot[2] < 0  # Negative angular velocity due to negative v
