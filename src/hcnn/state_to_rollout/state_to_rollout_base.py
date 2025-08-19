from abc import ABC, abstractmethod

import jax
from flax import nnx, struct

from hcnn.projection import Rollout


@struct.dataclass
class StateToRolloutBase(nnx.Module, ABC):
    state_dim: int
    action_dim: int
    constr_dim: int
    horizon: int

    @abstractmethod
    def __call__(self, state: jax.Array) -> Rollout:
        """Estimates the rollout given the current state.

        Args:
            state (jax.Array): Current state, shape (..., state_dim)

        Returns:
            Estimated rollout with states and actions
                states, shape (..., horizon, state_dim)
                actions, shape (..., horizon, action_dim)
        """
        pass
