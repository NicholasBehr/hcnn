#!/usr/bin/env python3
"""Test script for SystemDynamicsContinuous batch dimension handling."""

import jax
import jax.numpy as jnp

from src.hcnn.system_dynamics.system_dynamics_continuous import (
    SystemDynamicsContinuous,
)


class SimpleOscillator(SystemDynamicsContinuous):
    """Simple harmonic oscillator for testing."""

    def continuous_dynamics(
        self, state: jax.Array, action: jax.Array
    ) -> jax.Array:
        """Simple oscillator dynamics: dx/dt = [x[1], -x[0] + u]"""
        # state = [position, velocity], action = [force]
        x, v = state[..., 0], state[..., 1]
        u = action[..., 0] if action.shape[-1] > 0 else 0.0

        dx = v
        dv = -x + u

        return jnp.stack([dx, dv], axis=-1)


def test_batch_dimensions():
    """Test various batch dimensions."""
    print("Testing SystemDynamicsContinuous with batch dimensions...")

    # Create dynamics instance
    dynamics = SimpleOscillator.factory(state_dim=2, action_dim=1, dt=0.01)

    # Test 1: Single sample
    print("\n1. Single sample:")
    state = jnp.array([1.0, 0.0])  # shape: (2,)
    action = jnp.array([0.1])  # shape: (1,)
    next_state = dynamics(state, action)
    print(f"   Input state shape: {state.shape}")
    print(f"   Input action shape: {action.shape}")
    print(f"   Output state shape: {next_state.shape}")
    print(f"   Expected: (2,), Got: {next_state.shape}")
    assert next_state.shape == (2,), f"Expected (2,), got {next_state.shape}"

    # Test 2: Batch of samples
    print("\n2. Batch of samples:")
    state = jnp.array([[1.0, 0.0], [2.0, 1.0], [0.5, -0.5]])  # shape: (3, 2)
    action = jnp.array([[0.1], [0.2], [0.0]])  # shape: (3, 1)
    next_state = dynamics(state, action)
    print(f"   Input state shape: {state.shape}")
    print(f"   Input action shape: {action.shape}")
    print(f"   Output state shape: {next_state.shape}")
    print(f"   Expected: (3, 2), Got: {next_state.shape}")
    assert next_state.shape == (3, 2), (
        f"Expected (3, 2), got {next_state.shape}"
    )

    # Test 3: Higher dimensional batches
    print("\n3. Higher dimensional batches:")
    state = jnp.ones((2, 3, 2))  # shape: (2, 3, 2)
    action = jnp.ones((2, 3, 1))  # shape: (2, 3, 1)
    next_state = dynamics(state, action)
    print(f"   Input state shape: {state.shape}")
    print(f"   Input action shape: {action.shape}")
    print(f"   Output state shape: {next_state.shape}")
    print(f"   Expected: (2, 3, 2), Got: {next_state.shape}")
    assert next_state.shape == (2, 3, 2), (
        f"Expected (2, 3, 2), got {next_state.shape}"
    )

    print("\n✅ All batch dimension tests passed!")


if __name__ == "__main__":
    test_batch_dimensions()
