from abc import abstractmethod
from enum import Enum

import diffrax
import jax
from flax import struct

from hcnn.system_dynamics.system_dynamics_base import SystemDynamicsBase


class Integrator(Enum):
    Euler = diffrax.Euler()
    Tsit5 = diffrax.Tsit5()
    Dopri5 = diffrax.Dopri5()


@struct.dataclass
class SystemDynamicsContinuous(SystemDynamicsBase):
    """
    System dynamics for continuous-time systems of the form dx/dt = f(x, u).

    This class integrates the continuous dynamics using diffrax ODE solvers
    over a fixed timestep to compute the next discrete state.
    """

    dt: float
    integrator: Integrator = struct.field(
        pytree_node=False, default=Integrator.Dopri5
    )

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

    def _integrate_dynamics(
        self, state: jax.Array, action: jax.Array
    ) -> jax.Array:
        """Integrates continuous dynamics using diffrax.

        Args:
            state: Current state, shape (..., state_dim)
            action: Action (assumed constant over integration step),
                shape (..., action_dim)

        Returns:
            Next state after integration, shape (..., state_dim)
        """

        # Create ODE term
        term = diffrax.ODETerm(lambda t, x, u: self._continuous_dynamics(x, u))

        # Set up integration
        t0 = 0.0
        t1 = self.dt
        y0 = state

        # Solve ODE
        # Use DirectAdjoint to allow both forward and reverse mode autodiff
        solution = diffrax.diffeqsolve(
            term,
            self.integrator.value,
            t0=t0,
            t1=t1,
            dt0=self.dt,
            y0=y0,
            args=action,
            saveat=diffrax.SaveAt(t1=True),
            adjoint=diffrax.DirectAdjoint(),
        )

        # Return the final state, preserving batch dimensions
        # solution.ys has shape (1, ..., state_dim), we want (..., state_dim)
        if solution.ys is None:
            # Integration failed TODO: Handle this case more gracefully
            return y0
        return solution.ys[0]

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """Computes the next state by integrating continuous dynamics.

        Args:
            state: Current state, shape (..., state_dim)
            action: Action, shape (..., action_dim)

        Returns:
            Next state after integration, shape (..., state_dim)
        """
        return self._integrate_dynamics(state, action)
