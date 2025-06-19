import jax
import jax.numpy as jnp
from flax import struct

from hcnn.system_constraints import SystemConstraintsBase


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
        # Concatenate bounds
        lower_bound = jnp.concatenate([state_l, action_l])
        upper_bound = jnp.concatenate([state_u, action_u])
        # Find indices where the constraint is not always satisfied
        mask = ~((lower_bound == -jnp.inf) & (upper_bound == jnp.inf))
        active_indices = jnp.nonzero(mask)[0]
        # Filter bounds
        filtered_lower = lower_bound[active_indices]
        filtered_upper = upper_bound[active_indices]
        return cls(
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
