import jax
from flax import struct

from hcnn.optimization.optimization import OptimizationSolver
from hcnn.optimization.solver_qsqp import SolverQSQP

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
    solver: OptimizationSolver

    @staticmethod
    def factory(system: System, policy: Policy, constraint: Constraint) -> "Environment":
        problem = ProblemPolicyDeviation.factory(
            system=system,
            policy=policy,
            constraint=constraint,
            horizon=15,
            rho=30,
        )
        solver = SolverQSQP(atol=1e-3)
        return Environment(
            system=system,
            policy=policy,
            constraint=constraint,
            problem=problem,
            solver=solver,
        )

    @jax.jit
    def step(self, state: jax.Array) -> jax.Array:
        problem = self.problem.copy_with(initial_state=state)

        # Get the action from projection
        x_star = self.solver.solve(problem)
        action = problem.get_action(x_star)

        # Apply the action to the system
        next_state = self.system.step(state, action)
        return next_state
