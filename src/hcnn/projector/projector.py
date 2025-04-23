import jax
import jax.numpy as jnp
from flax import struct

from hcnn.constraint.constraint import Constraint
from hcnn.policy.policy import Policy
from hcnn.system.system import System


@struct.dataclass
class OptimizationVariable:
    actions: jax.Array
    policy_actions: jax.Array
    states: jax.Array
    relaxations: jax.Array
    max_relaxation: float
    rho: float  # no gradient in cost! TODO


@struct.dataclass
class Projector:
    system: System
    policy: Policy
    constraint: Constraint
    horizon: int

    def warm_start(self, state: jax.Array) -> OptimizationVariable:
        # allocate space
        actions = jax.numpy.zeros((self.horizon, self.system.action_dim))
        states = jax.numpy.zeros((self.horizon, self.system.state_dim))
        relaxations = jax.numpy.zeros((self.horizon, self.constraint.lower_bound.shape[0]))

        # fill in the first state
        states = states.at[0].set(state)

        for i in range(0, self.horizon):
            state = states[i]
            action = self.policy.get_action(state)
            next_state = self.system.step(state, action)

            actions = actions.at[i].set(action)
            if i < self.horizon - 1:
                states = states.at[i + 1].set(next_state)
            sa_constraint = self.constraint.function(state, action)
            relaxations = relaxations.at[i].set(
                jnp.maximum(0, self.constraint.lower_bound - sa_constraint)
                - jnp.maximum(0, sa_constraint - self.constraint.upper_bound)
            )

        return OptimizationVariable(
            actions=actions,
            policy_actions=actions,
            states=states,
            relaxations=relaxations,
            max_relaxation=float(jnp.max(jnp.abs(relaxations))),
            rho=10.0,
        )

    ### COST FUNCTIONS ###
    @staticmethod
    def cost_matvec_quadratic(x: OptimizationVariable) -> OptimizationVariable:
        return OptimizationVariable(
            actions=x.actions - x.policy_actions,
            policy_actions=x.policy_actions - x.actions,
            states=jnp.zeros_like(x.states),
            relaxations=jnp.zeros_like(x.relaxations),
            max_relaxation=0.0,
            rho=0.0,
        )

    @staticmethod
    def cost_linear(x: OptimizationVariable) -> OptimizationVariable:
        return OptimizationVariable(
            actions=jnp.zeros_like(x.actions),
            policy_actions=jnp.zeros_like(x.policy_actions),
            states=jnp.zeros_like(x.states),
            relaxations=jnp.zeros_like(x.relaxations),
            max_relaxation=jax.lax.stop_gradient(x.rho / 2),
            rho=0.0,
        )

    ### CONSTRAINT FUNCTIONS ###
    # (1) Initial state constraint
    def bounds_initial(self):
        lower_initial = jnp.zeros(self.system.state_dim)
        upper_initial = jnp.zeros(self.system.state_dim)
        return (lower_initial, upper_initial)

    @staticmethod
    def constraint_initial(x: OptimizationVariable, initial_state: jax.Array) -> jax.Array:
        return x.states[0] - initial_state

    # (2) Dynamics constraints
    def bounds_dynamics(self):
        lower_dynamics = jax.numpy.zeros((self.horizon - 1, self.system.state_dim))
        upper_dynamics = jax.numpy.zeros((self.horizon - 1, self.system.state_dim))
        return (lower_dynamics, upper_dynamics)

    def constraint_dynamics(self, x: OptimizationVariable) -> jax.Array:
        return jax.vmap(
            lambda state, action, next_state: next_state - self.system.step(state, action)
        )(x.states[:-1], x.actions[:-1], x.states[1:])

    # (3) Policy constraints
    def bounds_policy(self):
        lower_policy = jax.numpy.zeros((self.horizon, self.system.action_dim))
        upper_policy = jax.numpy.zeros((self.horizon, self.system.action_dim))
        return (lower_policy, upper_policy)

    def constraint_policy(self, x: OptimizationVariable) -> jax.Array:
        return jax.vmap(lambda state, policy_action: policy_action - self.policy.get_action(state))(
            x.states, x.policy_actions
        )

    # (4) State & action constraints
    def bounds_state_action(self):
        lower_sa = jax.numpy.tile(jnp.atleast_2d(self.constraint.lower_bound), self.horizon)
        upper_sa = jax.numpy.tile(jnp.atleast_2d(self.constraint.upper_bound), self.horizon)
        return (lower_sa, upper_sa)

    def constraint_state_action(self, x: OptimizationVariable) -> jax.Array:
        return jax.vmap(lambda state, action: self.constraint.function(state, action))(
            x.states, x.actions
        )

    # (5) Max relaxation constraints
    def bounds_relaxation(self):
        lower_relaxation = jnp.concatenate(
            (
                -jnp.inf * jnp.ones((self.horizon * self.constraint.lower_bound.shape[0],)),
                jnp.zeros((self.horizon * self.constraint.lower_bound.shape[0],)),
                0,
            )
        )
        upper_relaxation = jnp.concatenate(
            (
                jnp.zeros((self.horizon * self.constraint.upper_bound.shape[0],)),
                jnp.inf * jnp.zeros((self.horizon * self.constraint.upper_bound.shape[0],)),
                jnp.inf,
            )
        )
        return (lower_relaxation, upper_relaxation)

    def constraint_relaxation(self, x: OptimizationVariable) -> jax.Array:
        relax_positive = jax.vmap(lambda relaxation: relaxation - x.max_relaxation)(x.relaxations)
        relax_negative = jax.vmap(lambda relaxation: relaxation + x.max_relaxation)(x.relaxations)
        relax_max = x.max_relaxation
        return jnp.concatenate((relax_positive.flatten(), relax_negative.flatten(), relax_max))
