import jax
import jax.numpy as jnp
from flax import nnx, struct
from jax import flatten_util

from hcnn.policy import PolicyBase
from hcnn.projection.projection_base import (
    ProjectionBase,
    ProjectionPhiD,
    ProjectionPhiND,
    Rollout,
)
from hcnn.system_constraints import SystemConstraintsBase
from hcnn.system_dynamics import SystemDynamicsBase
from hcnn.training.dropout_mlp import DropoutMLP


@struct.dataclass
class ProjectionNoSubsNoTerm(ProjectionBase):
    @classmethod
    def factory(  # noqa: PLR0913
        cls,
        policy: PolicyBase,
        relax_to_go: DropoutMLP,
        sys_dynamics: SystemDynamicsBase,
        sys_constraints: SystemConstraintsBase,
        horizon: int,
        td_gamma: float,
        relaxation_penalty: float,
    ):
        state_dim = sys_dynamics.state_dim
        action_dim = sys_dynamics.action_dim
        constraint_dim = sys_constraints.constraint_dim

        # OptimizationProblemBase
        discount = (td_gamma ** jnp.arange(horizon))[:, None]
        cost_quad_diag = flatten_util.ravel_pytree(
            Rollout(
                states=jnp.zeros((horizon, state_dim)),
                actions=jnp.zeros((horizon, action_dim)),
                interventions=2 * jnp.ones((horizon, action_dim)) * discount,
                relaxations_plus=jnp.zeros((horizon, constraint_dim)),
                relaxations_minus=jnp.zeros((horizon, constraint_dim)),
                relaxation_terminal=jnp.zeros((1,)),
            )
        )[0]
        cost_linear = flatten_util.ravel_pytree(
            Rollout(
                states=jnp.zeros((horizon, state_dim)),
                actions=jnp.zeros((horizon, action_dim)),
                interventions=jnp.zeros((horizon, action_dim)),
                relaxations_plus=relaxation_penalty
                * jnp.ones((horizon, constraint_dim))
                * discount,
                relaxations_minus=relaxation_penalty
                * jnp.ones((horizon, constraint_dim))
                * discount,
                relaxation_terminal=jnp.zeros((1,)),
            )
        )[0]
        bound_tuple_list = [
            cls.bounds_initial_state(state_dim=state_dim),
            cls.bounds_sys_dynamics(horizon=horizon, state_dim=state_dim),
            cls.bounds_policy(horizon=horizon, action_dim=action_dim),
            cls.bounds_sys_constraints(
                horizon=horizon, sys_constraints=sys_constraints
            ),
            cls.bounds_non_neg_relax(
                horizon=horizon, constraint_dim=constraint_dim
            ),
        ]
        lower_bound_list, upper_bound_list = zip(
            *bound_tuple_list, strict=False
        )
        lower_bound = jnp.concatenate(lower_bound_list)
        upper_bound = jnp.concatenate(upper_bound_list)

        return ProjectionNoSubsNoTerm(
            cost_quad_diag=cost_quad_diag,
            cost_linear=cost_linear,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            state_dim=state_dim,
            action_dim=action_dim,
            constraint_dim=constraint_dim,
            horizon=horizon,
            td_gamma=td_gamma,
            relaxation_penalty=relaxation_penalty,
            policy_graphdef=nnx.split(policy)[0],
            relax_to_go_graphdef=nnx.split(relax_to_go)[0],
            sys_dynamics=sys_dynamics,
            sys_constraints=sys_constraints,
        )

    def constraint_fn(
        self, x: jax.Array, phi_d: ProjectionPhiD, phi_nd: ProjectionPhiND
    ) -> jax.Array:
        rollout = self.x_to_rollout(x)
        policy = nnx.merge(
            self.policy_graphdef, phi_d.policy_params, phi_nd.policy_non_params
        )

        constraints_list = [
            self.constraints_initial_state(
                rollout=rollout, initial_state=phi_nd.initial_state
            ),
            self.constraints_sys_dynamics(
                rollout=rollout, sys_dynamics=self.sys_dynamics
            ),
            self.constraints_policy(
                rollout=rollout, policy=policy, jax_key=phi_nd.jax_key
            ),
            self.constraints_sys_constraints(
                rollout=rollout, sys_constraints=self.sys_constraints
            ),
            self.constraints_non_neg_relax(rollout=rollout),
        ]
        return jnp.concatenate(constraints_list)
