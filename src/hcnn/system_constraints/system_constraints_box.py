import jax
import jax.numpy as jnp
from flax import struct

from .system_constraints_base import SystemConstraintsBase


@struct.dataclass
class SystemConstraintsBox(SystemConstraintsBase):
    # Indices of constraints that are not always satisfied
    active_indices: jax.Array

    @classmethod
    def factory(
        cls,
        state_l: jax.Array,
        state_u: jax.Array,
        action_l: jax.Array,
        action_u: jax.Array,
    ) -> "SystemConstraintsBox":
        # Get JAX's default float type to respect 32/64 bit config
        float_dtype = jnp.ones(1).dtype

        # Cast to floating respecting JAX config
        state_l = jnp.asarray(state_l, dtype=float_dtype)
        state_u = jnp.asarray(state_u, dtype=float_dtype)
        action_l = jnp.asarray(action_l, dtype=float_dtype)
        action_u = jnp.asarray(action_u, dtype=float_dtype)

        # Ensure bounds are in the correct order
        state_l = jnp.minimum(state_l, state_u)
        action_l = jnp.minimum(action_l, action_u)
        state_u = jnp.maximum(state_l, state_u)
        action_u = jnp.maximum(action_l, action_u)

        # Concatenate bounds
        lower_bound = jnp.concatenate([state_l, action_l])
        upper_bound = jnp.concatenate([state_u, action_u])

        # Find indices where the constraint is not always satisfied
        mask = ~((lower_bound == -jnp.inf) & (upper_bound == jnp.inf))
        active_indices = jnp.nonzero(mask)[0]

        # Filter bounds
        filtered_lower = lower_bound[active_indices]
        filtered_upper = upper_bound[active_indices]

        return SystemConstraintsBox(
            state_dim=state_l.shape[0],
            action_dim=action_l.shape[0],
            constraint_dim=active_indices.shape[0],
            lower_bound=filtered_lower,
            upper_bound=filtered_upper,
            active_indices=active_indices,
        )

    @jax.jit
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        full = jnp.concatenate([state, action], axis=-1)
        return full[..., self.active_indices]
