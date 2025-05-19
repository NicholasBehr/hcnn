import jax
from flax import nnx, struct

from .optimization import OptimizationSolverBase
from .policy import PolicyBase
from .projection import ProjectionBase


@struct.dataclass
class Environment:
    policy: PolicyBase
    projection: ProjectionBase
    solver: OptimizationSolverBase

    def step(self, state: jax.Array, *, jax_key: jax.Array) -> jax.Array:
        graphdef_pol, state_pol = nnx.split(self.policy)
        return self._step(
            graphdef_pol, state_pol, self.projection, self.solver, state, jax_key=jax_key
        )

    @staticmethod
    @jax.jit
    def _step(
        graphdef_pol,
        state_pol,
        projection,
        solver,
        state: jax.Array,
        *,
        jax_key: jax.Array,
    ) -> jax.Array:
        policy = nnx.merge(graphdef_pol, state_pol)
        policy_param = nnx.state(policy, nnx.Param)
        context = projection.Context(
            initial_state=state,
            policy_params=policy_param,
            jax_key=jax_key,
        )
        x_star = solver._solve(projection, context)
        rollout = projection.get_rollout(x_star)
        next_state = rollout.states[1]
        return next_state
