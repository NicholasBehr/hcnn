import jax
import jax.numpy as jnp
import pytest

from hcnn.system_dynamics import SystemDynamicsContinuous
from hcnn.system_dynamics.system_dynamics_continuous import Integrator

from .test_system_dynamics_base import TestSystemDynamicsBase


class SystemDynamicsContinuousMock(SystemDynamicsContinuous):
    """
    Mock implementation for testing with simple harmonic oscillator dynamics.
    """

    def _continuous_dynamics(
        self, state: jax.Array, action: jax.Array
    ) -> jax.Array:
        """Simple harmonic oscillator: dx/dt = [v, -x + u]

        Args:
            state: [position, velocity, ...], shape (..., state_dim)
            action: [force, ...], shape (..., action_dim)

        Returns:
            State derivative, shape (..., state_dim)
        """
        # For state_dim >= 2: position and velocity dynamics
        # For state_dim > 2: additional states have zero dynamics
        position = state[..., 0]
        velocity = state[..., 1] if state.shape[-1] > 1 else 0.0
        force = action[..., 0] if action.shape[-1] > 0 else 0.0

        # Simple harmonic oscillator with damping and external force
        dposition = velocity
        dvelocity = -position - 0.1 * velocity + force

        # Stack derivatives
        if state.shape[-1] == 1:
            return jnp.expand_dims(dposition, axis=-1)
        elif state.shape[-1] == 2:
            return jnp.stack([dposition, dvelocity], axis=-1)
        else:
            # For higher dimensions, additional states have zero dynamics
            zero_dynamics = jnp.zeros_like(state[..., 2:])
            return jnp.concatenate(
                [jnp.stack([dposition, dvelocity], axis=-1), zero_dynamics],
                axis=-1,
            )


class TestSystemDynamicsContinuous(TestSystemDynamicsBase):
    """Test class for SystemDynamicsContinuous."""

    @pytest.fixture
    def sys_dynamics(self) -> SystemDynamicsContinuous:
        return SystemDynamicsContinuousMock(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            dt=0.01,
            integrator=Integrator.Dopri5,
        )

    def test_instance_creation_continuous(
        self, sys_dynamics: SystemDynamicsContinuous
    ):
        assert isinstance(sys_dynamics, SystemDynamicsContinuous)
        assert sys_dynamics.state_dim == self.state_dim
        assert sys_dynamics.action_dim == self.action_dim
        assert isinstance(sys_dynamics.dt, float)
        assert sys_dynamics.dt == 0.01
        assert isinstance(sys_dynamics.integrator, Integrator)

    def test_continuous_dynamics_method(
        self, sys_dynamics: SystemDynamicsContinuous
    ):
        """Test the continuous_dynamics method directly."""
        key = jax.random.key(0)
        state = jax.random.normal(key, (sys_dynamics.state_dim,))
        action = jax.random.normal(key, (sys_dynamics.action_dim,))

        derivative = sys_dynamics._continuous_dynamics(state, action)
        assert derivative.shape == state.shape
        assert derivative.shape == (sys_dynamics.state_dim,)

    def test_integration_single_step(self):
        """Test integration produces reasonable results for a single step."""
        dynamics = SystemDynamicsContinuousMock(
            state_dim=2,
            action_dim=1,
            dt=0.001,  # Small step for accuracy
            integrator=Integrator.Dopri5,
        )

        # Start at [1.0, 0.0] (position=1, velocity=0)
        state = jnp.array([1.0, 0.0])
        action = jnp.array([0.0])  # No external force

        next_state = dynamics(state, action)

        # For harmonic oscillator, position should decrease slightly,
        # velocity should become negative
        assert next_state[0] < state[0]  # Position decreases
        assert next_state[1] < 0.0  # Velocity becomes negative

    def test_different_integrators(self):
        """Test that different integrators work."""
        integrators = [
            Integrator.Euler,
            Integrator.Tsit5,
            Integrator.Dopri5,
        ]

        state = jnp.array([1.0, 0.0])
        action = jnp.array([0.0])

        results = []
        for integrator in integrators:
            dynamics = SystemDynamicsContinuousMock(
                state_dim=2,
                action_dim=1,
                dt=0.01,
                integrator=integrator,
            )
            next_state = dynamics(state, action)
            results.append(next_state)

            # All should produce finite results
            assert jnp.all(jnp.isfinite(next_state))

        # Results should be similar but not identical due to different methods
        for i in range(1, len(results)):
            assert jnp.allclose(results[0], results[i], rtol=0.1)

    def test_zero_action_dynamics(self):
        """Test dynamics with zero action."""
        dynamics = SystemDynamicsContinuousMock(
            state_dim=2,
            action_dim=1,
            dt=0.01,
            integrator=Integrator.Dopri5,
        )

        state = jnp.array([1.0, 0.0])
        action = jnp.array([0.0])

        next_state = dynamics(state, action)

        # Should be different from initial state (system evolves)
        assert not jnp.allclose(next_state, state)
        assert jnp.all(jnp.isfinite(next_state))

    def test_with_action_influence(self):
        """Test that actions influence the dynamics."""
        dynamics = SystemDynamicsContinuousMock(
            state_dim=2,
            action_dim=1,
            dt=0.01,
            integrator=Integrator.Dopri5,
        )

        state = jnp.array([1.0, 0.0])
        action_zero = jnp.array([0.0])
        action_positive = jnp.array([1.0])

        next_state_zero = dynamics(state, action_zero)
        next_state_positive = dynamics(state, action_positive)

        # Different actions should produce different results
        assert not jnp.allclose(next_state_zero, next_state_positive)

    def test_analytical_solution_exponential_decay(self):
        """Test against known analytical solution for exponential decay."""

        class ExponentialDecayMock(SystemDynamicsContinuous):
            """Simple exponential decay: dx/dt = -k*x with k=2.0"""

            def _continuous_dynamics(
                self, state: jax.Array, action: jax.Array
            ) -> jax.Array:
                # Exponential decay: dx/dt = -k*x with k=2.0
                k = 2.0
                return -k * state

        # Test parameters
        k = 2.0
        dt = 0.01
        x0 = 3.0  # Initial condition

        dynamics = ExponentialDecayMock(
            state_dim=1,
            action_dim=1,
            dt=dt,
        )

        state = jnp.array([x0])
        action = jnp.array([0.0])  # No action needed for this system

        # Integrate one step
        next_state = dynamics(state, action)

        # Analytical solution: x(t) = x0 * exp(-k*t)
        analytical_solution = x0 * jnp.exp(-k * dt)

        # Check that numerical solution matches analytical solution
        # Use relative tolerance since we're comparing integration vs exact
        assert jnp.allclose(
            next_state[0],
            analytical_solution,
            rtol=1e-4,  # 0.01% relative tolerance
            atol=1e-6,  # Small absolute tolerance
        ), f"Numerical: {next_state[0]}, Analytical: {analytical_solution}"

        # Test multiple steps to verify cumulative accuracy
        current_state = state
        for i in range(10):
            current_state = dynamics(current_state, action)
            t = (i + 1) * dt
            analytical = x0 * jnp.exp(-k * t)

            # Allow slightly larger tolerance for longer integration
            assert jnp.allclose(
                current_state[0],
                analytical,
                rtol=1e-3,  # 0.1% relative tolerance
                atol=1e-5,
            ), (
                f"Step {i + 1}: Numerical: {current_state[0]}, "
                f"Analytical: {analytical}"
            )
