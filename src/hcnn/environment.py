import jax
from flax import struct

from .policy.policy import Policy
from .system.system import System


@struct.dataclass
class Environment:
    system: System
    policy: Policy

    @jax.jit
    def step(self, state: jax.Array) -> jax.Array:
        # Get the action from the policy
        action = self.policy.get_action(state)

        # Apply the action to the system
        next_state = self.system.step(state, action)
        return next_state
