import jax
import jax.numpy as jnp
from flax import struct

from .system_constraints_base import SystemConstraintsBase


@struct.dataclass
class SystemConstraintsMulti(SystemConstraintsBase):
    """Compose multiple system constraints into a single aggregate.

    This class wraps multiple ``SystemConstraintsBase`` implementations and
    exposes them as a single constraint by concatenating:

    - their outputs g_i(state, action) along the last axis, and
    - their bounds [lower_i, upper_i] along the constraint dimension.

    Contract
    - Inputs: ``state`` (..., state_dim), ``action`` (..., action_dim)
    - Output: concatenated constraints (..., sum_i constraint_dim_i)
        - Satisfaction: for all i,
            lower_bound[i] <= g(state, action)[i] <= upper_bound[i]
    """

    # Keep the list of child constraints as static metadata (non-pytree field)
    constraints: tuple[SystemConstraintsBase, ...] = struct.field(
        pytree_node=False
    )

    @classmethod
    def factory(
        cls,
        constraints: list[SystemConstraintsBase]
        | tuple[SystemConstraintsBase, ...],
    ) -> "SystemConstraintsMulti":
        """Create a multi-constraint aggregator from existing constraints.

        Args:
                constraints: A non-empty sequence of ``SystemConstraintsBase``
                        instances that share the same ``state_dim`` and
                        ``action_dim``.

        Returns:
                SystemConstraintsMulti with concatenated bounds and outputs.

        Raises:
                ValueError: If the sequence is empty or dimensions are
                inconsistent.
        """
        if not constraints:
            raise ValueError("constraints must be a non-empty sequence")

        # Ensure consistent state/action dimensions across all components
        state_dim = constraints[0].state_dim
        action_dim = constraints[0].action_dim
        for c in constraints:
            if c.state_dim != state_dim or c.action_dim != action_dim:
                raise ValueError(
                    "All constraints must share the same state_dim and"
                    " action_dim"
                )

        # Concatenate bounds along the constraint dimension
        lower = jnp.concatenate([c.lower_bound for c in constraints], axis=0)
        upper = jnp.concatenate([c.upper_bound for c in constraints], axis=0)

        return cls(
            state_dim=int(state_dim),
            action_dim=int(action_dim),
            constraint_dim=int(lower.shape[0]),
            lower_bound=lower,
            upper_bound=upper,
            constraints=tuple(constraints),
        )

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """Evaluate and concatenate all child constraints.

        Args:
                state: Array with shape (..., state_dim)
                action: Array with shape (..., action_dim)

        Returns:
                Array with shape (..., constraint_dim) equal to the
                concatenation of each child constraint's output along the
                last axis.
        """
        parts = [c(state, action) for c in self.constraints]
        # Fast-path for single constraint (still safe for JAX transforms)
        if len(parts) == 1:
            return parts[0]
        return jnp.concatenate(parts, axis=-1)
