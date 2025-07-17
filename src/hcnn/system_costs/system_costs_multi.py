from typing import Tuple

import jax
from flax import struct
from jax import numpy as jnp

from hcnn.system_costs.system_costs_base import SystemCostsBase


@struct.dataclass
class SystemCostsMulti(SystemCostsBase):
    costs: Tuple[SystemCostsBase, ...]

    @classmethod
    def factory(cls, *costs: SystemCostsBase):
        if not costs:
            raise ValueError("At least one cost must be provided.")
        # Check all costs have the same state_dim and action_dim
        state_dim = costs[0].state_dim
        action_dim = costs[0].action_dim
        for c in costs:
            if c.state_dim != state_dim or c.action_dim != action_dim:
                raise ValueError(
                    "All costs must have the same state_dim and action_dim."
                )
        return cls(
            state_dim=state_dim,
            action_dim=action_dim,
            costs=tuple(costs),
        )

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        total_cost = jnp.zeros(state.shape[:-1], dtype=state.dtype)
        for cost in self.costs:
            total_cost = total_cost + cost(state, action)
        return total_cost
