import jax
import jax.numpy as jnp
from flax import struct

from hcnn.constraint.constraint import Constraint
from hcnn.policy.policy import Policy
from hcnn.qcnc.qcnc import OptimizationProblem, OptimizationVariable
from hcnn.system.system import System


@struct.dataclass
class OptimizationVariableProjector(OptimizationVariable):
    actions: jax.Array
    policy_actions: jax.Array
    states: jax.Array


@struct.dataclass
class OptimizationProblemProjector(OptimizationProblem):
    system: System
    policy: Policy
    constraint: Constraint
    horizon: int
    initial_state: jax.Array

    def warm_start(self) -> OptimizationVariable:
        # allocate space
        actions = jax.numpy.zeros((self.horizon, self.system.action_dim))
        states = jax.numpy.zeros((self.horizon, self.system.state_dim))
        ineq_relaxations = jax.numpy.zeros((self.horizon, self.constraint.lower_bound.shape[0]))

        # fill in the first state
        states = states.at[0].set(self.initial_state)

        for i in range(0, self.horizon):
            state = states[i]
            action = self.policy.get_action(state)
            next_state = self.system.step(state, action)

            actions = actions.at[i].set(action)
            if i < self.horizon - 1:
                states = states.at[i + 1].set(next_state)
            ineq_constraint = self.constraint.function(state, action)
            ineq_relaxations = ineq_relaxations.at[i].set(
                jnp.maximum(0, self.constraint.lower_bound - ineq_constraint)
                - jnp.maximum(0, ineq_constraint - self.constraint.upper_bound)
            )

        return OptimizationVariableProjector(
            max_ineq_relaxation=float(jnp.max(jnp.abs(ineq_relaxations))),
            ineq_relaxations=ineq_relaxations,
            actions=actions,
            policy_actions=actions,
            states=states,
        )

    ### COST FUNCTIONS ###
    @staticmethod
    def cost_matvec_quadratic(x: OptimizationVariableProjector) -> OptimizationVariableProjector:
        return OptimizationVariableProjector(
            max_ineq_relaxation=0,
            ineq_relaxations=jnp.zeros_like(x.ineq_relaxations),
            actions=x.actions - x.policy_actions,
            policy_actions=x.policy_actions - x.actions,
            states=jnp.zeros_like(x.states),
        )

    @staticmethod
    def cost_linear(x: OptimizationVariableProjector, rho: float) -> OptimizationVariableProjector:
        return OptimizationVariableProjector(
            max_ineq_relaxation=rho / 2,
            ineq_relaxations=jnp.zeros_like(x.ineq_relaxations),
            actions=jnp.zeros_like(x.actions),
            policy_actions=jnp.zeros_like(x.policy_actions),
            states=jnp.zeros_like(x.states),
        )

    ### EQUALITY CONSTRAINT FUNCTIONS h(x) = 0 ###
    # (1) Initial state constraint
    def constraint_initial(self, x: OptimizationVariable) -> jax.Array:
        return x.states[0] - self.initial_state

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
    def constraints_equality(self, x: OptimizationVariable) -> jax.Array:
        constraint_initial = self.constraint_initial(x)
        constraint_dynamics = self.constraint_dynamics(x)
        constraint_policy = self.constraint_policy(x)

        return jnp.concatenate((constraint_initial, constraint_dynamics, constraint_policy))

    def bound_equality(self, x: OptimizationVariable) -> jax.Array:
        out = self.constraints_equality(x)
        return jnp.zeros_like(out)

    ### INEQUALITY CONSTRAINT FUNCTIONS l <= g(x) <= u ###
    def constraints_inequality(self, x: OptimizationVariable) -> jax.Array:
        return jax.vmap(lambda state, action: self.constraint.function(state, action))(
            x.states, x.actions
        ).ravel()

    def bounds_inequality(self):
        lower_sa = jnp.tile(self.constraint.lower_bound, self.horizon)
        upper_sa = jnp.tile(self.constraint.upper_bound, self.horizon)
        return (lower_sa, upper_sa)
