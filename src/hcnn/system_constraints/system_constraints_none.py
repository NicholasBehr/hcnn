import jax
import jax.numpy as jnp
from flax import struct

from .system_constraints_base import SystemConstraintsBase


@struct.dataclass
class SystemConstraintsNone(SystemConstraintsBase):
    """System constraints representing an unconstrained system."""

    @classmethod
    def factory(
        cls,
        state_dim: int,
        action_dim: int,
    ) -> "SystemConstraintsNone":
        """
        Create a SystemConstraintsNone instance representing no constraints.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.

        Returns:
            SystemConstraintsNone: An instance with no constraints.
        """
        return cls(
            state_dim=state_dim,
            action_dim=action_dim,
            constraint_dim=1,
            lower_bound=-jnp.ones(1),
            upper_bound=+jnp.ones(1),
        )

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """
        Evaluates the constraint function for an unconstrained system.

        Args:
            state (jax.Array): State, shape (..., state_dim).
            action (jax.Array): Action, shape (..., action_dim).

        Returns:
            jax.Array: Constraint array, shape (..., 1).
        """
        # Return an array with the correct batch dimensions and constraint_dim=1
        batch_shape = jnp.broadcast_shapes(state.shape[:-1], action.shape[:-1])
        return jnp.zeros(batch_shape + (1,))
