import jax
import jax.numpy as jnp
from flax import struct

from .system_constraints_base import SystemConstraintsBase


@struct.dataclass
class SystemConstraintsBall(SystemConstraintsBase):
    """Hypersphere exclusion constraint in state space.

    This nonconvex constraint excludes a ball (hypersphere) region centered at
    ``center`` with radius ``radius``. The constraint is satisfied when the
    state lies outside or on the boundary of the ball:

        g(state, action) = ||state - center||^2 - radius^2 >= 0.

    Notes
    - Only depends on ``state``; ``action`` is ignored.
    - ``constraint_dim = 1`` with bounds [0, +inf].
    - Does not depend on actions, but we still carry ``action_dim`` so
        this constraint can be composed with others. Provide it via the
        factory.
    """

    center: jax.Array  # shape (state_dim,)
    radius: jax.Array  # scalar array for dtype consistency

    @classmethod
    def factory(
        cls,
        center: jax.Array,
        radius: float | jax.Array,
        action_dim: int,
    ) -> "SystemConstraintsBall":
        """Create a hypersphere exclusion constraint.

        Args:
            center: Array of shape (state_dim,) specifying the sphere center.
            radius: Non-negative radius (float or 0-D/1-D array with one
                value).
            action_dim: The action space dimension used for composition.

        Returns:
            SystemConstraintsBall instance with lower bound 0 and upper
            bound +inf.

        Raises:
            ValueError: If ``radius`` is negative or ``center`` has invalid
            shape.
        """
        # Respect JAX default float dtype (32/64 depending on config)
        float_dtype = jnp.ones(1).dtype

        center = jnp.asarray(center, dtype=float_dtype)
        if center.ndim != 1:
            raise ValueError("center must be a 1D array of shape (state_dim,)")

        radius_arr = jnp.asarray(radius, dtype=float_dtype)
        radius_arr = jnp.reshape(radius_arr, ())  # ensure scalar array
        if (radius_arr < 0).item():
            raise ValueError("radius must be non-negative")

        state_dim = int(center.shape[0])
        if not isinstance(action_dim, int) or action_dim < 0:
            raise ValueError("action_dim must be a non-negative integer")

        lower = jnp.asarray([0.0], dtype=float_dtype)
        upper = jnp.asarray([jnp.inf], dtype=float_dtype)

        return cls(
            state_dim=state_dim,
            action_dim=action_dim,
            constraint_dim=1,
            lower_bound=lower,
            upper_bound=upper,
            center=center,
            radius=radius_arr,
        )

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """Evaluate the exclusion constraint value.

        Args:
            state: Array with shape (..., state_dim)
            action: Ignored; can be any shape with matching leading batch dims.

        Returns:
            Array with shape (..., 1): ||state - center||^2 - radius^2.
        """
        diff = state - self.center
        val = jnp.sum(diff * diff, axis=-1, keepdims=True) - (
            self.radius * self.radius
        )
        return val
