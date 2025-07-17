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
        # Get JAX's default float type to respect 32/64 bit config
        float_dtype = jnp.ones(1).dtype

        # For an unconstrained system, there are no constraints
        return cls(
            state_dim=state_dim,
            action_dim=action_dim,
            constraint_dim=0,
            lower_bound=jnp.array([], dtype=float_dtype),
            upper_bound=jnp.array([], dtype=float_dtype),
        )

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """
        Evaluates the constraint function for an unconstrained system.

        Args:
            state (jax.Array): State, shape (..., state_dim).
            action (jax.Array): Action, shape (..., action_dim).

        Returns:
            jax.Array: Empty constraint array, shape (..., 0).
        """
        # Return an empty array with the correct batch dimensions
        batch_shape = jnp.broadcast_shapes(state.shape[:-1], action.shape[:-1])
        return jnp.empty(batch_shape + (0,), dtype=state.dtype)
