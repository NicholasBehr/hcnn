from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx, struct
from jax import flatten_util

from hcnn.optimization import OptimizationProblemBase
from hcnn.policy import PolicyBase
from hcnn.system_constraints import SystemConstraintsBase
from hcnn.system_dynamics import SystemDynamicsBase
from hcnn.training import DropoutMLP


@struct.dataclass
class Rollout:
    states: jax.Array  # Shape: (horizon, state_dim)
    actions: jax.Array  # Shape: (horizon, action_dim)
    interventions: jax.Array  # Shape: (horizon, action_dim)
    relaxations_plus: jax.Array  # Shape: (horizon, constraint_dim)
    relaxations_minus: jax.Array  # Shape: (horizon, constraint_dim)
    relaxation_terminal: jax.Array  # Shape: (1,)


@struct.dataclass
class ProjectionPhiD:
    policy_params: nnx.State
    relax_to_go_params: nnx.State


@struct.dataclass
class ProjectionPhiND:
    policy_non_params: nnx.State
    relax_to_go_non_params: nnx.State
    initial_state: jax.Array  # Shape: (state_dim,)
    jax_key: jax.Array


@struct.dataclass
class ProjectionBase(OptimizationProblemBase[ProjectionPhiD, ProjectionPhiND]):
    state_dim: int = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    constraint_dim: int = struct.field(pytree_node=False)
    horizon: int = struct.field(pytree_node=False)
    relaxation_penalty: float = struct.field(pytree_node=False)
    td_gamma: float = struct.field(pytree_node=False)

    policy_graphdef: nnx.GraphDef
    relax_to_go_graphdef: nnx.GraphDef
    sys_dynamics: SystemDynamicsBase
    sys_constraints: SystemConstraintsBase

    def x_to_rollout(self, x: jax.Array) -> Rollout:
        template = Rollout(
            actions=jnp.empty((self.horizon, self.action_dim)),
            states=jnp.empty((self.horizon, self.state_dim)),
            interventions=jnp.empty((self.horizon, self.action_dim)),
            relaxations_plus=jnp.empty((self.horizon, self.constraint_dim)),
            relaxations_minus=jnp.empty((self.horizon, self.constraint_dim)),
            relaxation_terminal=jnp.empty((1,)),
        )
        _, unravel_pytree = flatten_util.ravel_pytree(template)
        return unravel_pytree(x)

    def get_x0(
        self, phi_d: ProjectionPhiD, phi_nd: ProjectionPhiND
    ) -> jax.Array:
        def next_state_action_row(state, _):
            policy = nnx.merge(
                self.policy_graphdef,
                phi_d.policy_params,
                phi_nd.policy_non_params,
            )
            action = policy(state, jax_key=phi_nd.jax_key)
            next_state = self.sys_dynamics(state, action)
            return next_state, (action, state)

        _, (actions, states) = jax.lax.scan(
            next_state_action_row,
            init=phi_nd.initial_state,
            length=self.horizon,
        )

        constraint_values = self.sys_constraints(states, actions)
        relaxations_plus = jnp.maximum(
            0.0, self.sys_constraints.lower_bound - constraint_values
        )
        relaxations_minus = jnp.maximum(
            0.0, constraint_values - self.sys_constraints.upper_bound
        )
        relaxation_to_go = nnx.merge(
            self.relax_to_go_graphdef,
            phi_d.relax_to_go_params,
            phi_nd.relax_to_go_non_params,
        )
        relaxation_terminal = relaxation_to_go(
            states[-1], rngs=nnx.Rngs(phi_nd.jax_key)
        )
        rollout = Rollout(
            states=states,
            actions=actions,
            interventions=jnp.zeros((self.horizon, self.action_dim)),
            relaxations_plus=relaxations_plus,
            relaxations_minus=relaxations_minus,
            relaxation_terminal=relaxation_terminal,
        )

        return flatten_util.ravel_pytree(rollout)[0]

    # (1) initial state
    @staticmethod
    def constraints_initial_state(
        rollout: Rollout, initial_state: jax.Array
    ) -> jax.Array:
        return rollout.states[0] - initial_state

    @staticmethod
    def bounds_initial_state(state_dim) -> Tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros(state_dim),
            jnp.zeros(state_dim),
        )

    # (2) system dynamics
    @staticmethod
    def constraints_sys_dynamics(
        rollout: Rollout, sys_dynamics: SystemDynamicsBase
    ) -> jax.Array:
        states_i = rollout.states[:-1]
        states_i_plus_1 = rollout.states[1:]
        actions_i = rollout.actions[:-1]

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

    # (3) policy
    @staticmethod
    def constraints_policy(
        rollout: Rollout, policy: PolicyBase, jax_key: jax.Array
    ) -> jax.Array:
        constraints = (
            rollout.actions
            - policy(rollout.states, jax_key=jax_key)
            - rollout.interventions
        )
        return constraints.ravel()

    @staticmethod
    def bounds_policy(
        horizon: int, action_dim: int
    ) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros(action_dim * horizon),
            jnp.zeros(action_dim * horizon),
        )

    # (4) system constraints
    @staticmethod
    def constraints_sys_constraints(
        rollout: Rollout, sys_constraints: SystemConstraintsBase
    ) -> jax.Array:
        constraints = jnp.array(
            (
                sys_constraints(rollout.states, rollout.actions)
                + rollout.relaxations_plus
                - rollout.relaxations_minus
            )
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

    # (5) non-negative relaxations
    @staticmethod
    def constraints_non_neg_relax(
        rollout: Rollout,
    ) -> jax.Array:
        return jnp.concatenate(
            [rollout.relaxations_plus, rollout.relaxations_minus]
        ).ravel()

    @staticmethod
    def bounds_non_neg_relax(
        horizon: int,
        constraint_dim: int,
    ) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros(2 * horizon * constraint_dim),
            jnp.ones(2 * horizon * constraint_dim) * jnp.inf,
        )

    # (6) terminal relaxation
    @staticmethod
    def constraints_terminal_relax(
        rollout: Rollout, relax_to_go: DropoutMLP, jax_key: jax.Array
    ) -> jax.Array:
        return rollout.relaxation_terminal - relax_to_go(
            rollout.states[-1], rngs=nnx.Rngs(jax_key)
        )

    @staticmethod
    def bounds_terminal_relax() -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros(1),
            jnp.zeros(1),
        )
