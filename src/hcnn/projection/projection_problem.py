import jax
import jax.numpy as jnp
from flax import struct
from jax import flatten_util

from hcnn.optimization import OptimizationProblemBase
from hcnn.system_constraints import SystemConstraintsBase
from hcnn.system_dynamics import SystemDynamicsBase


@struct.dataclass
class OptimizationVariable:
    # projected trajecory
    states: jax.Array  # shape (horizon, state_dim)
    actions: jax.Array  # shape (horizon, action_dim)
    # projection distance
    states_diff: jax.Array  # shape (horizon, state_dim)
    actions_diff: jax.Array  # shape (horizon, action_dim)
    # state relaxation for ineq. constraint satisfaction
    # action constraints are not to be relaxed
    relax_pos: jax.Array  # shape (horizon, constr_dim)
    relax_neg: jax.Array  # shape (horizon, constr_dim)


@struct.dataclass
class ProjectionPhiD:
    states_ref: jax.Array  # shape (horizon, state_dim)
    actions_ref: jax.Array  # shape (horizon, action_dim)


@struct.dataclass
class ProjectionPhiND:
    # for constraints
    sys_dynamics: SystemDynamicsBase
    initial_state: jax.Array
    # for warm start
    relax_pos: jax.Array  # shape (horizon, constr_dim)
    relax_neg: jax.Array  # shape (horizon, constr_dim)


@struct.dataclass
class ProjectionProblem(
    OptimizationProblemBase[
        ProjectionPhiD,
        ProjectionPhiND,
    ]
):
    # system constraints and horizon are FIXED for the duration of the jit
    sys_constraints: SystemConstraintsBase = struct.field(pytree_node=False)
    horizon: int = struct.field(pytree_node=False)

    @classmethod
    def factory(
        cls,
        sys_constraints: SystemConstraintsBase,
        horizon: int,
        relaxation_penalty: float,
    ):
        state_dim = sys_constraints.state_dim
        action_dim = sys_constraints.action_dim
        constr_dim = sys_constraints.constraint_dim

        states_one = jnp.ones((horizon, state_dim))
        actions_one = jnp.ones((horizon, action_dim))
        constr_one = jnp.ones((horizon, constr_dim))

        cost_quad_diag = flatten_util.ravel_pytree(
            OptimizationVariable(
                states=0 * states_one,
                actions=0 * actions_one,
                states_diff=2 * states_one,
                actions_diff=2 * actions_one,
                relax_pos=2 * constr_one,
                relax_neg=2 * constr_one,
            )
        )[0]
        cost_linear = flatten_util.ravel_pytree(
            OptimizationVariable(
                states=0 * states_one,
                actions=0 * actions_one,
                states_diff=0 * states_one,
                actions_diff=0 * actions_one,
                relax_pos=relaxation_penalty * constr_one,
                relax_neg=relaxation_penalty * constr_one,
            )
        )[0]
        bound_tuple_list = [
            cls.bounds_rollout_diff(
                horizon=horizon,
                state_dim=state_dim,
                action_dim=action_dim,
            ),
            cls.bounds_sys_dynamics(
                horizon=horizon,
                state_dim=state_dim,
            ),
            cls.bounds_sys_constraints(
                horizon=horizon,
                sys_constraints=sys_constraints,
            ),
            cls.bounds_initial_state(
                state_dim=state_dim,
            ),
            cls.bounds_relaxation(
                horizon=horizon,
                constr_dim=constr_dim,
            ),
        ]
        lower_bound_list, upper_bound_list = zip(
            *bound_tuple_list, strict=False
        )
        lower_bound = jnp.concatenate(lower_bound_list)
        upper_bound = jnp.concatenate(upper_bound_list)

        return cls(
            cost_quad_diag=cost_quad_diag,
            cost_linear=cost_linear,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            sys_constraints=sys_constraints,
            horizon=horizon,
        )

    ## ROLLOUT DIFFERENCE CONSTRAINTS ###
    @staticmethod
    def constraints_rollout_diff(
        optvar: OptimizationVariable, phi_d: ProjectionPhiD
    ) -> jax.Array:
        state_diff = optvar.states - phi_d.states_ref - optvar.states_diff
        action_diff = optvar.actions - phi_d.actions_ref - optvar.actions_diff
        return jnp.concatenate([state_diff.ravel(), action_diff.ravel()])

    @staticmethod
    def bounds_rollout_diff(
        horizon: int, state_dim: int, action_dim: int
    ) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros(horizon * (state_dim + action_dim)),
            jnp.zeros(horizon * (state_dim + action_dim)),
        )

    ## SYSTEM DYNAMICS CONSTRAINTS ###
    @staticmethod
    def constraints_sys_dynamics(
        optvar: OptimizationVariable, sys_dynamics: SystemDynamicsBase
    ) -> jax.Array:
        states_i = optvar.states[:-1]
        states_i_plus_1 = optvar.states[1:]
        actions_i = optvar.actions[:-1]

        constraints = states_i_plus_1 - sys_dynamics(states_i, actions_i)
        return constraints.ravel()

    @staticmethod
    def bounds_sys_dynamics(
        horizon: int, state_dim: int
    ) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros((horizon - 1) * state_dim),
            jnp.zeros((horizon - 1) * state_dim),
        )

    ## SYSTEM CONSTRAINTS ###
    @staticmethod
    def constraints_sys_constraints(
        optvar: OptimizationVariable, sys_constraints: SystemConstraintsBase
    ) -> jax.Array:
        constraints = (
            sys_constraints(optvar.states, optvar.actions)
            + optvar.relax_pos
            - optvar.relax_neg
        )
        return constraints.ravel()

    @staticmethod
    def bounds_sys_constraints(
        horizon: int, sys_constraints: SystemConstraintsBase
    ) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.repeat(sys_constraints.lower_bound, horizon),
            jnp.repeat(sys_constraints.upper_bound, horizon),
        )

    ## INITIAL STATE CONSTRAINTS ###
    @staticmethod
    def constraints_initial_state(
        optvar: OptimizationVariable, phi_nd: ProjectionPhiND
    ) -> jax.Array:
        return optvar.states[0] - phi_nd.initial_state

    @staticmethod
    def bounds_initial_state(state_dim: int) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros(state_dim),
            jnp.zeros(state_dim),
        )

    ## POSITIVE RELAXATION CONSTRAINTS ##
    @staticmethod
    def constraints_relaxation(optvar: OptimizationVariable) -> jax.Array:
        return jnp.concatenate(
            [optvar.relax_pos.ravel(), optvar.relax_neg.ravel()]
        )

    @staticmethod
    def bounds_relaxation(
        horizon: int, constr_dim: int
    ) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.repeat(0, horizon * 2 * constr_dim),
            jnp.repeat(jnp.inf, horizon * 2 * constr_dim),
        )

    def get_x0(
        self, phi_d: ProjectionPhiD, phi_nd: ProjectionPhiND
    ) -> jax.Array:
        optvar = OptimizationVariable(
            states=phi_d.states_ref,
            actions=phi_d.actions_ref,
            states_diff=jnp.zeros_like(phi_d.states_ref),
            actions_diff=jnp.zeros_like(phi_d.actions_ref),
            relax_pos=phi_nd.relax_pos,
            relax_neg=phi_nd.relax_neg,
        )
        return flatten_util.ravel_pytree(optvar)[0]

    def x_to_optvar(self, x: jax.Array) -> OptimizationVariable:
        state_dim = self.sys_constraints.state_dim
        action_dim = self.sys_constraints.action_dim
        constr_dim = self.sys_constraints.constraint_dim
        state_template = jnp.empty((self.horizon, state_dim))
        action_template = jnp.empty((self.horizon, action_dim))
        constr_template = jnp.empty((self.horizon, constr_dim))

        optvar_template = OptimizationVariable(
            states=state_template,
            actions=action_template,
            states_diff=state_template,
            actions_diff=action_template,
            relax_pos=constr_template,
            relax_neg=constr_template,
        )
        _, unravel_pytree = flatten_util.ravel_pytree(optvar_template)
        return unravel_pytree(x)

    def constraint_fn(self, x, phi_d, phi_nd):
        optvar = self.x_to_optvar(x)
        constraints_list = [
            self.constraints_rollout_diff(
                optvar=optvar,
                phi_d=phi_d,
            ),
            self.constraints_sys_dynamics(
                optvar=optvar,
                sys_dynamics=phi_nd.sys_dynamics,
            ),
            self.constraints_sys_constraints(
                optvar=optvar,
                sys_constraints=self.sys_constraints,
            ),
            self.constraints_initial_state(
                optvar=optvar,
                phi_nd=phi_nd,
            ),
            self.constraints_relaxation(
                optvar=optvar,
            ),
        ]
        return jnp.concatenate(constraints_list)
