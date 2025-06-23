from abc import abstractmethod
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx, struct

from hcnn.optimization import OptimizationProblemBase
from hcnn.policy import PolicyBase
from hcnn.system_constraints import SystemConstraintsBase
from hcnn.system_dynamics import SystemDynamicsBase


@struct.dataclass
class Rollout:
    states: jax.Array
    actions: jax.Array
    interventions: jax.Array
    relaxations_plus: jax.Array
    relaxations_minus: jax.Array


@struct.dataclass
class ProjectionPhi:
    initial_state: jax.Array
    policy_state: nnx.State
    relax_to_go_state: nnx.State
    jax_key: jax.Array = struct.field(pytree_node=False)


@struct.dataclass
class ProjectionBase(OptimizationProblemBase[ProjectionPhi]):
    state_dim: int = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    constraint_dim: int = struct.field(pytree_node=False)
    horizon: int = struct.field(pytree_node=False)

    policy_graphdef: nnx.GraphDef
    relax_to_go_graphdef: nnx.GraphDef
    sys_dynamics: SystemDynamicsBase
    sys_constraints: SystemConstraintsBase

    @jax.jit
    @abstractmethod
    def x_to_rollout(self, x: jax.Array) -> Rollout:
        pass

    # (1) initial state
    @staticmethod
    def constraints_initial_state(
        rollout: Rollout, initial_state: jax.Array
    ) -> jax.Array:
        return rollout.states[0] - initial_state

    @staticmethod
    def bounds_initial_state(state_dim) -> Tuple[jax.Array, jax.Array]:
        return (
            jax.numpy.zeros(state_dim),
            jax.numpy.zeros(state_dim),
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
