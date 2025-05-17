import jax
from flax import nnx, struct

from .optimization import OptimizationSolverBase
from .policy import PolicyBase
from .projection import ProjectionBase
from .system_dynamics import SystemDynamicsBase


@struct.dataclass
class Environment:
    sys_dynamics: SystemDynamicsBase
    policy: PolicyBase
    projection: ProjectionBase
    solver: OptimizationSolverBase

    def step(self, state: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        policy_param = nnx.state(self.policy, nnx.Param)
        problem = self.projection.get_problem(state, policy_param, rngs=rngs)
        x_star = self.solver.solve(problem)
        rollout = self.projection.get_rollout(x_star)

        # Apply the action to the system
        next_state = self.sys_dynamics.step(state, rollout.actions[0])
        return next_state
