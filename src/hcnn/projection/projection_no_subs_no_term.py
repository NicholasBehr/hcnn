from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx, struct
from jax import flatten_util

from hcnn.policy import PolicyBase
from hcnn.system_constraints import SystemConstraintsBase
from hcnn.system_dynamics import SystemDynamicsBase

from .projection_base import ProjectionBase, RolloutPlan


@struct.dataclass
class ProjectionNoSubstitutionNoTerminal(ProjectionBase):
    @struct.dataclass
    class OptimizationVariable:
        actions: jax.Array
        states: jax.Array
        interventions: jax.Array
        relaxations_plus: jax.Array
        relaxations_minus: jax.Array

    get_optvar: Callable[[jax.Array], OptimizationVariable] = struct.field(pytree_node=False)

    @classmethod
    def factory(
        cls,
        sys_dynamics: SystemDynamicsBase,
        sys_constraints: SystemConstraintsBase,
        policy: PolicyBase,
        horizon: int,
        relaxation_penalty: float,
    ) -> "ProjectionNoSubstitutionNoTerminal":
        cost_quad_diag, get_optvar = flatten_util.ravel_pytree(
            cls.OptimizationVariable(
                actions=jnp.zeros((horizon, sys_dynamics.action_dim)),
                states=jnp.zeros((horizon, sys_dynamics.state_dim)),
                interventions=2 * jnp.ones(horizon),
                relaxations_plus=jnp.zeros(horizon),
                relaxations_minus=jnp.zeros(horizon),
            )
        )
        cost_linear, _ = flatten_util.ravel_pytree(
            cls.OptimizationVariable(
                actions=jnp.zeros((horizon, sys_dynamics.action_dim)),
                states=jnp.zeros((horizon, sys_dynamics.state_dim)),
                interventions=jnp.zeros(horizon),
                relaxations_plus=relaxation_penalty * jnp.ones(horizon),
                relaxations_minus=relaxation_penalty * jnp.ones(horizon),
            )
        )

        bound_tuple_list = [
            cls._bounds_initial_state(sys_dynamics),
            cls._bounds_sys_dynamics(horizon, sys_dynamics),
            cls._bounds_policy(horizon, sys_dynamics),
            cls._bounds_sys_constraints(horizon, sys_constraints),
            cls._bounds_non_neg_relax(horizon),
        ]
        lower_bound_list, upper_bound_list = zip(*bound_tuple_list, strict=False)
        lower_bound = jnp.concatenate(lower_bound_list)
        upper_bound = jnp.concatenate(upper_bound_list)

        graph, _, other = nnx.split(policy, nnx.Param, ...)

        return ProjectionNoSubstitutionNoTerminal(
            sys_dynamics=sys_dynamics,
            sys_constraints=sys_constraints,
            horizon=horizon,
            relaxation_penalty=relaxation_penalty,
            cost_quad_diag=cost_quad_diag,
            cost_linear=cost_linear,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            get_optvar=get_optvar,
            policy_graph=graph,
            policy_other=other,
        )

    # TODO carry?
    def get_x0(
        self,
        initial_state: jax.Array,
        policy_params: nnx.State,
        *,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        ## define shapes
        actions = jnp.zeros((self.horizon, self.sys_dynamics.action_dim))
        states = jnp.zeros((self.horizon, self.sys_dynamics.state_dim))
        interventions = jnp.zeros(self.horizon)
        relaxations_plus = jnp.zeros(self.horizon)
        relaxations_minus = jnp.zeros(self.horizon)

        # fill in the first state
        states = states.at[0].set(initial_state)

        policy = nnx.merge(self.policy_graph, policy_params, self.policy_other)

        # propagate actions, states and relaxations (no interventions)
        for i in range(0, self.horizon):
            state = states[i]
            action = policy(state, rngs=nnx.clone(rngs))
            next_state = self.sys_dynamics.step(state, action)

            actions = actions.at[i].set(action)
            if i < self.horizon - 1:
                states = states.at[i + 1].set(next_state)
            ineq_constraint = self.sys_constraints.__call__(state, action)
            relaxations_plus = relaxations_plus.at[i].set(
                jnp.max(jnp.maximum(0.0, self.sys_constraints.lower_bound - ineq_constraint)),
            )
            relaxations_minus = relaxations_minus.at[i].set(
                jnp.max(jnp.maximum(0.0, ineq_constraint - self.sys_constraints.upper_bound)),
            )

        optvar = self.OptimizationVariable(
            actions=actions,
            states=states,
            interventions=interventions,
            relaxations_plus=relaxations_plus,
            relaxations_minus=relaxations_minus,
        )

        return flatten_util.ravel_pytree(optvar)[0]

    def get_h(
        self,
        initial_state: jax.Array,
        x: jax.Array,
        policy_params: nnx.State,
        *,
        rngs: nnx.Rngs,
    ):
        optvar = self.get_optvar(x)
        constraint_fun_list = [
            self._fun_initial_state(optvar, initial_state),
            self._constraints_sys_dynamics(optvar),
            self._constraints_policy(optvar, policy_params, rngs),
            self._constraints_sys_constraints(optvar),
            self._constraints_non_neg_relax(optvar),
        ]
        return jnp.concatenate(constraint_fun_list)

    def get_rollout(
        self,
        x_star: jax.Array,
    ) -> RolloutPlan:
        optvar = self.get_optvar(x_star)
        return RolloutPlan(
            states=optvar.states,
            actions=optvar.actions,
            interventions=optvar.interventions,
            relaxations=jnp.maximum(optvar.relaxations_plus, optvar.relaxations_minus),
        )

    # (1) initial state
    def _fun_initial_state(
        self, optvar: OptimizationVariable, initial_state: jax.Array
    ) -> jax.Array:
        return optvar.states[0] - initial_state

    @staticmethod
    def _bounds_initial_state(sys_dynamics) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros(sys_dynamics.state_dim),
            jnp.zeros(sys_dynamics.state_dim),
        )

    # (2) system dynamics
    def _constraints_sys_dynamics(self, optvar: OptimizationVariable) -> jax.Array:
        return jax.vmap(
            lambda state, action, next_state: next_state - self.sys_dynamics.step(state, action)
        )(optvar.states[:-1], optvar.actions[:-1], optvar.states[1:]).ravel()

    @staticmethod
    def _bounds_sys_dynamics(horizon, sys_dynamics) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros((horizon - 1) * sys_dynamics.state_dim),
            jnp.zeros((horizon - 1) * sys_dynamics.state_dim),
        )

    # (3) policy
    def _constraints_policy(self, optvar: OptimizationVariable, policy_params, rngs) -> jax.Array:
        policy = nnx.merge(self.policy_graph, policy_params, self.policy_other)

        @nnx.vmap(in_axes=(0, None, 0, 0, None), out_axes=0)
        def constraint(action, policy, state, intervention, rngs):
            return action - policy(state, rngs=rngs) - intervention

        return constraint(
            optvar.actions,
            policy,
            optvar.states,
            optvar.interventions,
            nnx.clone(rngs),
        ).ravel()

    @staticmethod
    def _bounds_policy(horizon, sys_dynamics) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros(sys_dynamics.action_dim * horizon),
            jnp.zeros(sys_dynamics.action_dim * horizon),
        )

    # (4) system constraints
    def _constraints_sys_constraints(self, optvar: OptimizationVariable) -> jax.Array:
        return jax.vmap(
            lambda state, action, relaxation_plus, relaxation_minus: self.sys_constraints(
                state, action
            )
            + relaxation_plus
            - relaxation_minus
        )(optvar.states, optvar.actions, optvar.relaxations_plus, optvar.relaxations_minus).ravel()

    @staticmethod
    def _bounds_sys_constraints(horizon, sys_constraints) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.tile(sys_constraints.lower_bound, horizon),
            jnp.tile(sys_constraints.upper_bound, horizon),
        )

    # (5) non-negative relaxations
    def _constraints_non_neg_relax(self, optvar: OptimizationVariable) -> jax.Array:
        return jnp.concatenate((optvar.relaxations_plus, optvar.relaxations_minus))

    @staticmethod
    def _bounds_non_neg_relax(horizon) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros(2 * horizon),
            jnp.ones(2 * horizon) * jnp.inf,
        )
