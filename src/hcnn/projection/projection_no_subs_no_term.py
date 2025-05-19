from functools import partial
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

    @partial(jax.jit, static_argnames=["horizon"])
    def get_x0(
        self,
        horizon: int,
        initial_state: jax.Array,
        policy_params: nnx.State,
        *,
        jax_key: jax.Array,
    ) -> jax.Array:
        policy = nnx.merge(self.policy_graph, policy_params, self.policy_other)

        def next_optvar_row(state, i):
            action = policy(state, jax_key=jax.random.fold_in(jax_key, i))
            next_state = self.sys_dynamics.step(state, action)
            constraint = self.sys_constraints(state, action)
            intervention = jnp.array(0)
            relaxation_plus = jnp.max(
                jnp.maximum(0.0, self.sys_constraints.lower_bound - constraint)
            )
            relaxations_minus = jnp.max(
                jnp.max(jnp.maximum(0.0, constraint - self.sys_constraints.upper_bound)),
            )
            out = (action, state, intervention, relaxation_plus, relaxations_minus)
            return next_state, out

        (_, optvar_tuple) = jax.lax.scan(
            f=next_optvar_row,
            init=initial_state,
            xs=jnp.arange(horizon),
            unroll=True,
        )

        optvar = self.OptimizationVariable(
            actions=optvar_tuple[0],
            states=optvar_tuple[1],
            interventions=optvar_tuple[2],
            relaxations_plus=optvar_tuple[3],
            relaxations_minus=optvar_tuple[4],
        )

        return flatten_util.ravel_pytree(optvar)[0]

    @jax.jit
    def get_h(
        self,
        initial_state: jax.Array,
        x: jax.Array,
        policy_params: nnx.State,
        *,
        jax_key: jax.Array,
    ):
        optvar = self.get_optvar(x)
        constraint_fun_list = [
            self._fun_initial_state(optvar, initial_state),
            self._constraints_sys_dynamics(optvar),
            self._constraints_policy(optvar, policy_params, jax_key),
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
    def _constraints_policy(
        self, optvar: OptimizationVariable, policy_params, jax_key
    ) -> jax.Array:
        policy = nnx.merge(self.policy_graph, policy_params, self.policy_other)

        def constraint(action, state, intervention, i):
            return action - policy(state, jax_key=jax.random.fold_in(jax_key, i)) - intervention

        return jax.vmap(constraint)(
            optvar.actions,
            optvar.states,
            optvar.interventions,
            jnp.arange(len(optvar.interventions)),
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
