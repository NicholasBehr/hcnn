"""
Kinematic Bike Model System Dynamics

This module implements a kinematic bike model commonly used in control
applications. The model represents the vehicle as a bicycle with front and
rear axles, where the front wheels can be steered and the vehicle has a
single velocity.

Physical Model:
- Single-track (bicycle) model with front and rear axles
- Front wheel steering with steering angle δ
- Rear-wheel drive with longitudinal acceleration control
- No slip assumption (kinematic model)
- State vector: [x, y, θ, v] where:
  - x, y: position in global coordinates (m)
  - θ: heading angle from x-axis (rad)
  - v: velocity magnitude (m/s)
- Control inputs: [a, δ] where:
  - a: longitudinal acceleration (m/s²)
  - δ: front wheel steering angle (rad)

The kinematic bike model is widely used in path planning and control for
autonomous vehicles due to its simplicity and reasonable accuracy at
moderate speeds where tire slip can be neglected.
"""

import jax
import jax.numpy as jnp
from flax import struct

from hcnn.system_dynamics.system_dynamics_continuous import (
    SystemDynamicsContinuous,
)


@struct.dataclass
class SystemDynamicsBike(SystemDynamicsContinuous):
    """
    Kinematic bike model dynamics for vehicle control applications.

    This implements the standard single-track kinematic model where the vehicle
    is represented as a bicycle with controllable front wheel steering and
    rear wheel acceleration.

    Physical Parameters:
    - wheelbase: wheelbase length (distance between front and rear axles)
    - max_steering: maximum steering angle magnitude (rad)

    State representation: [x, y, θ, v]
    - x, y: position in global coordinates (m)
    - θ: heading angle from positive x-axis (rad)
    - v: forward velocity (m/s)

    Action: [a, δ]
    - a: longitudinal acceleration (m/s²)
    - δ: front wheel steering angle (rad)
    """

    # Physical parameters
    wheelbase: float = 2.5  # Wheelbase length (m) - typical car value
    max_steering: float = 0.5  # Maximum steering angle (rad) - ~28.6 degrees

    @classmethod
    def factory(
        cls,
        wheelbase: float = 2.5,
        max_steering: float = 0.5,
        dt: float = 0.1,
    ) -> "SystemDynamicsBike":
        """
        Factory method to create a kinematic bike model system.

        Args:
            wheelbase: Wheelbase length (m). Typical values: 2.5-3.0m for cars
            max_steering: Maximum steering angle magnitude (rad)
            dt: Integration timestep (s)

        Returns:
            SystemDynamicsBike instance with state_dim=4, action_dim=2

        Note:
            The kinematic model assumes no tire slip and is most accurate
            at moderate speeds where lateral forces are small.
        """
        # Validate physical parameters
        assert wheelbase > 0, "Wheelbase length must be positive"
        assert max_steering > 0, "Maximum steering angle must be positive"
        assert dt > 0, "Timestep must be positive"

        return cls(
            state_dim=4,  # [x, y, θ, v]
            action_dim=2,  # [a, δ] - acceleration and steering
            dt=dt,
            wheelbase=wheelbase,
            max_steering=max_steering,
        )

    def _continuous_dynamics(
        self, state: jax.Array, action: jax.Array
    ) -> jax.Array:
        """
        Compute the continuous dynamics dx/dt = f(x, u) for the kinematic
        bike model.

        The kinematic bike model equations are:
        ẋ = v * cos(θ)
        ẏ = v * sin(θ)
        θ̇ = (v / L) * tan(δ)
        v̇ = a

        where the steering angle δ is limited to prevent unrealistic turning
        radii.

        Args:
            state: [x, y, θ, v] - position, heading, and velocity
            action: [a, δ] - acceleration and steering angle

        Returns:
            State derivative [ẋ, ẏ, θ̇, v̇]
        """
        # Extract state variables
        theta = state[..., 2]  # Heading angle
        v = state[..., 3]  # Velocity

        # Extract control inputs
        a = action[..., 0]  # Acceleration
        delta = action[..., 1]  # Steering angle

        # Kinematic bike model equations
        x_dot = v * jnp.cos(theta)
        y_dot = v * jnp.sin(theta)
        theta_dot = (v / self.wheelbase) * jnp.tan(delta)
        v_dot = a

        # Stack the derivatives
        state_dot = jnp.stack([x_dot, y_dot, theta_dot, v_dot], axis=-1)

        return state_dot
