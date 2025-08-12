import jax
import jax.numpy as jnp
from flax import struct
from jax import flatten_util

from hcnn.optimization import OptimizationProblemBase
from hcnn.system_constraints import SystemConstraintsBase
from hcnn.system_dynamics import SystemDynamicsBase

from .rollout import Rollout


@struct.dataclass
class OptimizationVariable:
    rollout: Rollout
    rollout_diff: Rollout


@struct.dataclass
class ProjectionPhiD:
    rollout_ref: Rollout


@struct.dataclass
class ProjectionPhiND:
    sys_dynamics: SystemDynamicsBase
    initial_state: jax.Array


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
    ):
        state_dim = sys_constraints.state_dim
        action_dim = sys_constraints.action_dim

        two = Rollout(
            states=2 * jnp.ones((horizon, state_dim)),
            actions=2 * jnp.ones((horizon, action_dim)),
        )
        zeros = Rollout(
            states=jnp.zeros((horizon, state_dim)),
            actions=jnp.zeros((horizon, action_dim)),
        )

        cost_quad_diag = flatten_util.ravel_pytree(
            OptimizationVariable(
                rollout=zeros,
                rollout_diff=two,  # quadratic cost for difference
            )
        )[0]
        cost_linear = flatten_util.ravel_pytree(
            OptimizationVariable(
                rollout=zeros,
                rollout_diff=zeros,
            )
        )[0]
        bound_tuple_list = [
            cls.bounds_rollout_diff(horizon, state_dim, action_dim),
            cls.bounds_sys_dynamics(horizon, state_dim),
            cls.bounds_sys_constraints(horizon, sys_constraints),
            cls.bounds_initial_state(state_dim),
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

    ## ROLLOUT DIFFERENCE CONSTRAINTS (linear) ###
    @staticmethod
    def constraints_rollout_diff(
        optvar: OptimizationVariable, rollout_ref: Rollout
    ) -> jax.Array:
        flat_rollout = flatten_util.ravel_pytree(optvar.rollout)[0]
        flat_rollout_ref = flatten_util.ravel_pytree(rollout_ref)[0]
        flat_rollout_diff = flatten_util.ravel_pytree(optvar.rollout_diff)[0]
        return flat_rollout - flat_rollout_ref - flat_rollout_diff

    @staticmethod
    def bounds_rollout_diff(
        horizon: int, state_dim: int, action_dim: int
    ) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros(horizon * (state_dim + action_dim)),
            jnp.zeros(horizon * (state_dim + action_dim)),
        )

    ## SYSTEM DYNAMICS CONSTRAINTS (linear if system is linear) ###
    @staticmethod
    def constraints_sys_dynamics(
        optvar: OptimizationVariable, sys_dynamics: SystemDynamicsBase
    ) -> jax.Array:
        states_i = optvar.rollout.states[:-1]
        states_i_plus_1 = optvar.rollout.states[1:]
        actions_i = optvar.rollout.actions[:-1]

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
        constraints = sys_constraints(
            optvar.rollout.states, optvar.rollout.actions
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
        optvar: OptimizationVariable, initial_state: jax.Array
    ) -> jax.Array:
        return optvar.rollout.states[0] - initial_state

    @staticmethod
    def bounds_initial_state(state_dim: int) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros(state_dim),
            jnp.zeros(state_dim),
        )

    def get_x0(
        self, phi_d: ProjectionPhiD, phi_nd: ProjectionPhiND
    ) -> jax.Array:
        rollout_ref = phi_d.rollout_ref
        optvar = OptimizationVariable(
            rollout=rollout_ref,
            rollout_diff=Rollout(
                states=jnp.zeros_like(rollout_ref.states),
                actions=jnp.zeros_like(rollout_ref.actions),
            ),
        )
        return flatten_util.ravel_pytree(optvar)[0]

    def x_to_optvar(self, x: jax.Array) -> OptimizationVariable:
        state_dim = self.sys_constraints.state_dim
        action_dim = self.sys_constraints.action_dim
        template = Rollout(
            states=jnp.empty((self.horizon, state_dim)),
            actions=jnp.empty((self.horizon, action_dim)),
        )
        optvar = OptimizationVariable(
            rollout=template,
            rollout_diff=template,
        )
        _, unravel_pytree = flatten_util.ravel_pytree(optvar)
        return unravel_pytree(x)

    def constraint_fn(self, x, phi_d, phi_nd):
        optvar = self.x_to_optvar(x)
        rollout_ref = phi_d.rollout_ref
        sys_dynamics = phi_nd.sys_dynamics
        initial_state = phi_nd.initial_state

        constraints_list = [
            self.constraints_rollout_diff(optvar, rollout_ref),
            self.constraints_sys_dynamics(optvar, sys_dynamics),
            self.constraints_sys_constraints(optvar, self.sys_constraints),
            self.constraints_initial_state(optvar, initial_state),
        ]
        return jnp.concatenate(constraints_list)
