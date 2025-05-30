import jax
from flax import nnx, struct

from .optimization import OptimizationSolverBase
from .policy import PolicyBase
from .projection import ProjectionBase, ProjectionPhi


@struct.dataclass
class Environment:
    policy: PolicyBase
    projection: ProjectionBase
    solver: OptimizationSolverBase

    def step(self, state: jax.Array, *, jax_key: jax.Array) -> jax.Array:
        policy_param = nnx.state(self.policy, nnx.Param)
        context = ProjectionPhi(
            initial_state=state,
            policy_params=policy_param,
            jax_key=jax_key,
        )
        x_star = self.solver.solve(self.projection, context)
        rollout = self.projection.get_rollout(x_star)
        next_state = rollout.states[1]
        return next_state
