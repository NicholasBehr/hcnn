import jax
from flax import struct

from hcnn.optimization.solver_linearize import SolverLinearize

from .constraint.constraint import Constraint
from .optimization.problem_policy_deviation import ProblemPolicyDeviation
from .policy.policy import Policy
from .system.system import System


@struct.dataclass
class Environment:
    system: System
    policy: Policy
    constraint: Constraint  # state action constraint
    problem: ProblemPolicyDeviation

    @staticmethod
    def factory(system: System, policy: Policy, constraint: Constraint) -> "Environment":
        problem = ProblemPolicyDeviation.factory(
            system=system,
            policy=policy,
            constraint=constraint,
            horizon=5,
            rho=250,
        )
        return Environment(system=system, policy=policy, constraint=constraint, problem=problem)

    @jax.jit
    def step(self, state: jax.Array) -> jax.Array:
        problem = self.problem.copy_with(initial_state=state)

        # Get the action from projection
        x_star = SolverLinearize.solve(problem)
        action = problem.get_action(x_star)

        # Apply the action to the system
        next_state = self.system.step(state, action)
        return next_state
