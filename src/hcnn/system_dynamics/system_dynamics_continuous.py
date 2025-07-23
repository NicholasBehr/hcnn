from abc import abstractmethod

import jax
from flax import struct

from hcnn.system_dynamics.system_dynamics_base import SystemDynamicsBase


@struct.dataclass
class SystemDynamicsContinuous(SystemDynamicsBase):
    """
    System dynamics for continuous-time systems of the form dx/dt = f(x, u).

    Integrates continuous dynamics using RK4 over a fixed timestep to compute
    the next discrete state. Supports arbitrary batch dimensions.
    """

    dt: float

    @abstractmethod
    def _continuous_dynamics(
        self, state: jax.Array, action: jax.Array
    ) -> jax.Array:
        """Abstract method for continuous dynamics dx/dt = f(x, u).

        Args:
            state: Current state, shape (..., state_dim)
            action: Action, shape (..., action_dim)

        Returns:
            State derivative dx/dt, shape (..., state_dim)
        """
        pass

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """Computes next state by integrating continuous dynamics with RK4.

        Args:
            state: Current state, shape (..., state_dim)
            action: Action (constant over integration), shape (..., action_dim)

        Returns:
            Next state after integration, shape (..., state_dim)
        """
        h = self.dt

        # RK4 integration
        k1 = self._continuous_dynamics(state, action)
        k2 = self._continuous_dynamics(state + h * k1 / 2, action)
        k3 = self._continuous_dynamics(state + h * k2 / 2, action)
        k4 = self._continuous_dynamics(state + h * k3, action)

        return state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
