import jax
from flax import nnx

from .policy.policy import Policy
from .state_action_constraint.state_action_constraint import StateActionConstraint
from .system.system import System


class Environment(nnx.Module):
    system: System
    state_action_constraint: StateActionConstraint
    policy: Policy
    horizon: int

    def __init__(
        self,
        system: System,
        policy: Policy,
        constraint: StateActionConstraint,
    ):
        self.system = system
        self.policy = policy
        self.state_action_constraint = constraint
        self.horizon = 5

    def cache_projector(self):
        pass

    def step(self, state: jax.Array) -> jax.Array:
        # problem = self.problem.copy_with(initial_state=state)

        # Get the action from projection
        # x_star = self.solver.solve(problem)
        # action = problem.get_action(x_star)
        action = self.policy(state)

        # Apply the action to the system
        next_state = self.system.step(state, action)
        return next_state
