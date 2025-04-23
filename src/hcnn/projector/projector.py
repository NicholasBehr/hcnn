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

    ### EQUALITY CONSTRAINT FUNCTIONS h(x) = 0 ###
    # (1) Initial state constraint
    @staticmethod
    def constraint_initial(x: OptimizationVariable, initial_state: jax.Array) -> jax.Array:
        return x.states[0] - initial_state

    # (2) Dynamics constraints
    def constraint_dynamics(self, x: OptimizationVariable) -> jax.Array:
        return jax.vmap(
            lambda state, action, next_state: next_state - self.system.step(state, action)
        )(x.states[:-1], x.actions[:-1], x.states[1:]).ravel()

    # (3) Policy constraints
    def constraint_policy(self, x: OptimizationVariable) -> jax.Array:
        return jax.vmap(lambda state, policy_action: policy_action - self.policy.get_action(state))(
            x.states, x.policy_actions
        ).ravel()

    # (4) Combination of all equality constraints
    def bounds_equality(self, x: OptimizationVariable):
        out = self.constraint_equality(x, x.states[0])
        return (jnp.zeros_like(out), jnp.zeros_like(out))

    def constraint_equality(self, x: OptimizationVariable, initial_state: jax.Array) -> jax.Array:
        constraint_initial = self.constraint_initial(x, initial_state)
        constraint_dynamics = self.constraint_dynamics(x)
        constraint_policy = self.constraint_policy(x)

        return jnp.concatenate((constraint_initial, constraint_dynamics, constraint_policy))

    ### INEQUALITY CONSTRAINT FUNCTIONS l <= g(x) <= u ###
    # (1) State & action constraints
    def bounds_state_action(self):
        lower_sa = jnp.tile(self.constraint.lower_bound, self.horizon)
        upper_sa = jnp.tile(self.constraint.upper_bound, self.horizon)
        return (lower_sa, upper_sa)

    def constraint_state_action(self, x: OptimizationVariable) -> jax.Array:
        return jax.vmap(lambda state, action: self.constraint.function(state, action))(
            x.states, x.actions
        ).ravel()

    # (2) Max relaxation constraints
    def bounds_relaxation(self):
        lower_relaxation = jnp.concatenate(
            (
                -jnp.inf * jnp.ones((self.horizon * self.constraint.lower_bound.shape[0],)),
                jnp.zeros((self.horizon * self.constraint.lower_bound.shape[0],)),
                jnp.zeros((1,)),
            )
        )
        upper_relaxation = jnp.concatenate(
            (
                jnp.zeros((self.horizon * self.constraint.upper_bound.shape[0],)),
                jnp.inf * jnp.ones((self.horizon * self.constraint.upper_bound.shape[0],)),
                jnp.inf * jnp.ones((1,)),
            )
        )
        return (lower_relaxation, upper_relaxation)

    def constraint_relaxation(self, x: OptimizationVariable) -> jax.Array:
        relax_positive = jax.vmap(lambda relaxation: relaxation - x.max_relaxation)(x.relaxations)
        relax_negative = jax.vmap(lambda relaxation: relaxation + x.max_relaxation)(x.relaxations)
        relax_max = jnp.array(x.max_relaxation)
        return jnp.concatenate((relax_positive.ravel(), relax_negative.ravel(), relax_max.ravel()))

    # (3) Combination of all inequality constraints
    def bounds_inequality(self):
        lower_sa, upper_sa = self.bounds_state_action()
        lower_relaxation, upper_relaxation = self.bounds_relaxation()

        return (
            jnp.concatenate((lower_sa, lower_relaxation)),
            jnp.concatenate((upper_sa, upper_relaxation)),
        )

    def constraint_inequality(self, x: OptimizationVariable) -> jax.Array:
        constraint_state_action = self.constraint_state_action(x)
        constraint_relaxation = self.constraint_relaxation(x)

        return jnp.concatenate((constraint_state_action, constraint_relaxation))
