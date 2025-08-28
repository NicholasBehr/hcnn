import jax
import jax.numpy as jnp
from flax import struct
from jax import flatten_util

from hcnn.solver import SQPProblemBase
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


@struct.dataclass
class ProjectionPhi:
    states_ref: jax.Array  # shape (horizon, state_dim)
    actions_ref: jax.Array  # shape (horizon, action_dim)
    initial_state: jax.Array  # shape (state_dim,)


@struct.dataclass
class ProjectionProblem(SQPProblemBase[ProjectionPhi]):
    sys_dynamics: SystemDynamicsBase
    sys_constraints: SystemConstraintsBase
    horizon: int = struct.field(pytree_node=False)

    @classmethod
    def factory(
        cls,
        sys_dynamics: SystemDynamicsBase,
        sys_constraints: SystemConstraintsBase,
        horizon: int,
    ):
        state_dim = sys_constraints.state_dim
        action_dim = sys_constraints.action_dim
        states_one = jnp.ones((horizon, state_dim))
        actions_one = jnp.ones((horizon, action_dim))

        cost_quad_diag = flatten_util.ravel_pytree(
            OptimizationVariable(
                states=0 * states_one,
                actions=0 * actions_one,
                states_diff=2 * states_one,
                actions_diff=2 * actions_one,
            )
        )[0]
        cost_linear = flatten_util.ravel_pytree(
            OptimizationVariable(
                states=0 * states_one,
                actions=0 * actions_one,
                states_diff=0 * states_one,
                actions_diff=0 * actions_one,
            )
        )[0]
        phi = ProjectionPhi(
            states_ref=0 * states_one,
            actions_ref=0 * actions_one,
            initial_state=0 * states_one[0],
        )
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
        ]
        lower_bound_list, upper_bound_list = zip(
            *bound_tuple_list, strict=False
        )
        lower_bound = jnp.concatenate(lower_bound_list)
        upper_bound = jnp.concatenate(upper_bound_list)

        return cls(
            cost_quad_diag=cost_quad_diag,
            cost_linear=cost_linear,
            phi=phi,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            sys_dynamics=sys_dynamics,
            sys_constraints=sys_constraints,
            horizon=horizon,
        )

    ## ROLLOUT DIFFERENCE CONSTRAINTS ###
    @staticmethod
    def constraints_rollout_diff(
        optvar: OptimizationVariable, phi: ProjectionPhi
    ) -> jax.Array:
        state_diff = optvar.states - phi.states_ref - optvar.states_diff
        action_diff = optvar.actions - phi.actions_ref - optvar.actions_diff
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
        return sys_constraints(optvar.states, optvar.actions).ravel()

    @staticmethod
    def bounds_sys_constraints(
        horizon: int, sys_constraints: SystemConstraintsBase
    ) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.tile(sys_constraints.lower_bound, horizon),
            jnp.tile(sys_constraints.upper_bound, horizon),
        )

    ## INITIAL STATE CONSTRAINTS ###
    @staticmethod
    def constraints_initial_state(
        optvar: OptimizationVariable, phi: ProjectionPhi
    ) -> jax.Array:
        return optvar.states[0] - phi.initial_state

    @staticmethod
    def bounds_initial_state(state_dim: int) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros(state_dim),
            jnp.zeros(state_dim),
        )

    def get_x0(self, phi: ProjectionPhi) -> jax.Array:
        state_dim = self.sys_constraints.state_dim
        action_dim = self.sys_constraints.action_dim
        states = jnp.empty((self.horizon, state_dim))
        actions = jnp.empty((self.horizon, action_dim))
        optvar = OptimizationVariable(
            states=phi.states_ref,
            actions=phi.actions_ref,
            states_diff=jnp.zeros_like(states),
            actions_diff=jnp.zeros_like(actions),
        )
        return flatten_util.ravel_pytree(optvar)[0]

    def x_to_optvar(self, x: jax.Array) -> OptimizationVariable:
        state_dim = self.sys_constraints.state_dim
        action_dim = self.sys_constraints.action_dim
        state_template = jnp.empty((self.horizon, state_dim))
        action_template = jnp.empty((self.horizon, action_dim))

        optvar_template = OptimizationVariable(
            states=jnp.empty_like(state_template),
            actions=jnp.empty_like(action_template),
            states_diff=jnp.empty_like(state_template),
            actions_diff=jnp.empty_like(action_template),
        )
        _, unravel_pytree = flatten_util.ravel_pytree(optvar_template)
        return unravel_pytree(x)

    def constraint_fn_phi(self, x, phi):
        optvar = self.x_to_optvar(x)
        constraints_list = [
            self.constraints_rollout_diff(
                optvar=optvar,
                phi=phi,
            ),
            self.constraints_sys_dynamics(
                optvar=optvar,
                sys_dynamics=self.sys_dynamics,
            ),
            self.constraints_sys_constraints(
                optvar=optvar,
                sys_constraints=self.sys_constraints,
            ),
            self.constraints_initial_state(
                optvar=optvar,
                phi=phi,
            ),
        ]
        return jnp.concatenate(constraints_list)
