import jax
import jax.numpy as jnp
from flax import nnx, struct
from jax import flatten_util

from hcnn.policy import PolicyBase
from hcnn.projection import ProjectionBase, ProjectionPhi, Rollout
from hcnn.system_constraints import SystemConstraintsBase
from hcnn.system_dynamics import SystemDynamicsBase


@struct.dataclass
class ProjectionNoSubstitutionNoTerminal(ProjectionBase):
    @classmethod
    def factory(
        cls,
        policy: PolicyBase,
        sys_dynamics: SystemDynamicsBase,
        sys_constraints: SystemConstraintsBase,
        horizon: int,
        relaxation_penalty: float,
    ):
        state_dim = sys_dynamics.state_dim
        action_dim = sys_dynamics.action_dim

        # OptimizationProblemBase
        cost_quad_diag = flatten_util.ravel_pytree(
            Rollout(
                actions=jnp.zeros((horizon, action_dim)),
                states=jnp.zeros((horizon, state_dim)),
                interventions=2 * jnp.ones((horizon, action_dim)),
                relaxations_plus=jnp.zeros(horizon),
                relaxations_minus=jnp.zeros(horizon),
            )
        )[0]
        cost_linear = flatten_util.ravel_pytree(
            Rollout(
                actions=jnp.zeros((horizon, action_dim)),
                states=jnp.zeros((horizon, state_dim)),
                interventions=jnp.zeros((horizon, action_dim)),
                relaxations_plus=relaxation_penalty * jnp.ones(horizon),
                relaxations_minus=relaxation_penalty * jnp.ones(horizon),
            )
        )[0]
        bound_tuple_list = [
            cls.bounds_initial_state(state_dim=state_dim),
            cls.bounds_sys_dynamics(horizon=horizon, state_dim=state_dim),
            cls.bounds_policy(horizon=horizon, action_dim=action_dim),
            cls.bounds_sys_constraints(
                horizon=horizon, sys_constraints=sys_constraints
            ),
            cls.bounds_non_neg_relax(horizon=horizon),
        ]
        lower_bound_list, upper_bound_list = zip(
            *bound_tuple_list, strict=False
        )
        lower_bound = jnp.concatenate(lower_bound_list)
        upper_bound = jnp.concatenate(upper_bound_list)

        # ProjectionBase
        policy_graphdef, _, policy_non_params = nnx.split(
            policy, nnx.Param, ...
        )
        split_dynamics = nnx.split(sys_dynamics)
        split_constraints = nnx.split(sys_constraints)

        return ProjectionNoSubstitutionNoTerminal(
            cost_quad_diag=cost_quad_diag,
            cost_linear=cost_linear,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            state_dim=state_dim,
            action_dim=action_dim,
            horizon=horizon,
            policy_graphdef=policy_graphdef,
            policy_non_params=policy_non_params,
            split_dynamics=split_dynamics,
            split_constraints=split_constraints,
        )

    @jax.jit
    def x_to_rollout(self, x: jax.Array) -> Rollout:
        template = Rollout(
            actions=jnp.empty((self.horizon, self.action_dim)),
            states=jnp.empty((self.horizon, self.state_dim)),
            interventions=jnp.empty((self.horizon, self.action_dim)),
            relaxations_plus=jnp.empty(self.horizon),
            relaxations_minus=jnp.empty(self.horizon),
        )
        _, unravel_pytree = flatten_util.ravel_pytree(template)
        return unravel_pytree(x)

    @jax.jit
    def get_x0(self, phi: ProjectionPhi) -> jax.Array:
        policy = nnx.merge(
            self.policy_graphdef, phi.policy_params, self.policy_non_params
        )
        sys_dynamics = nnx.merge(*self.split_dynamics)
        sys_constraints = nnx.merge(*self.split_constraints)

        @nnx.scan(
            in_axes=(None, None, None, nnx.Carry, None),
            out_axes=(nnx.Carry, 0),
            length=self.horizon,
        )
        def next_row(policy, sys_dynamics, sys_constraints, state, jax_key):
            action = policy(state, jax_key=jax_key)
            intervention = jnp.zeros_like(action)
            next_state = sys_dynamics(state, action)
            constraint = sys_constraints(state, action)
            relaxation_plus = jnp.max(
                jnp.maximum(0.0, sys_constraints.lower_bound - constraint)
            )
            relaxations_minus = jnp.max(
                jnp.maximum(0.0, constraint - sys_constraints.upper_bound)
            )
            out = (
                action,
                state,
                intervention,
                relaxation_plus,
                relaxations_minus,
            )
            return next_state, out

        _, rollout_tuple = next_row(
            policy=policy,
            sys_dynamics=sys_dynamics,
            sys_constraints=sys_constraints,
            state=phi.initial_state,
            jax_key=phi.jax_key,
        )

        rollout = Rollout(
            actions=rollout_tuple[0],
            states=rollout_tuple[1],
            interventions=rollout_tuple[2],
            relaxations_plus=rollout_tuple[3],
            relaxations_minus=rollout_tuple[4],
        )

        return flatten_util.ravel_pytree(rollout)[0]

    @jax.jit
    def constraint_fn(self, x: jax.Array, phi: ProjectionPhi) -> jax.Array:
        rollout = self.x_to_rollout(x)
        policy = nnx.merge(
            self.policy_graphdef, phi.policy_params, self.policy_non_params
        )
        sys_dynamics = nnx.merge(*self.split_dynamics)
        sys_constraints = nnx.merge(*self.split_constraints)

        constraints_list = [
            self.constraints_initial_state(
                rollout=rollout, initial_state=phi.initial_state
            ),
            self.constraints_sys_dynamics(
                rollout=rollout, sys_dynamics=sys_dynamics
            ),
            self.constraints_policy(
                rollout=rollout, policy=policy, jax_key=phi.jax_key
            ),
            self.constraints_sys_constraints(
                rollout=rollout, sys_constraints=sys_constraints
            ),
            self.constraints_non_neg_relax(rollout=rollout),
        ]
        return jnp.concatenate(constraints_list)
