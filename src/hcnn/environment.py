import jax

from .policy.policy import Policy
from .system.system import System


class Environment:
    system: System
    policy: Policy

    def __init__(self, system: System, policy: Policy):
        self.system = system
        self.policy = policy

    def step(self, state: jax.Array) -> jax.Array:
        # Get the action from the policy
        action = self.policy.get_action(state)

        # Apply the action to the system
        next_state = self.system.step(state, action)
        return next_state

    def vector_field(self):
        pass
