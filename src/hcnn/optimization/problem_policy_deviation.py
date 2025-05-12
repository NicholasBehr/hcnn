import jax
import jax.numpy as jnp
from flax import struct
from jax import flatten_util

from hcnn.optimization.optimization import OptimizationProblem
from hcnn.policy.policy import Policy
from hcnn.state_action_constraint.state_action_constraint import StateActionConstraint
from hcnn.system.system import System


@struct.dataclass
class ProblemPolicyDeviation(OptimizationProblem):
    @struct.dataclass
    class OptimizationVariable:
        actions: jax.Array
        policy_actions: jax.Array
        states: jax.Array
        relaxations: jax.Array
        abs_relaxations: jax.Array

    _system: System
    _policy: Policy
    _constraint: StateActionConstraint

    _initial_state: jax.Array
    _x0: OptimizationVariable

    @property
    def x0(self) -> jax.Array:
        return flatten_util.ravel_pytree(self._x0)[0]

    @staticmethod
    def factory(
        system: System,
        policy: Policy,
        constraint: StateActionConstraint,
        horizon: int,
        rho: float,
    ) -> "ProblemPolicyDeviation":
        ## define shapes
        actions = jax.numpy.zeros((horizon, system.action_dim))
        states = jax.numpy.zeros((horizon, system.state_dim))
        relaxations = jax.numpy.zeros((horizon, constraint.lower_bound.shape[0]))

        _x0 = ProblemPolicyDeviation.OptimizationVariable(
            actions=actions,
            policy_actions=actions,
            states=states,
            relaxations=relaxations,
            abs_relaxations=relaxations,
        )

        ## define linear cost 'c'
        cost_linear = flatten_util.ravel_pytree(
            ProblemPolicyDeviation.OptimizationVariable(
                actions=jnp.zeros_like(_x0.actions),
                policy_actions=jnp.zeros_like(_x0.policy_actions),
                states=jnp.zeros_like(_x0.states),
                relaxations=jnp.zeros_like(_x0.relaxations),
                abs_relaxations=rho / 2.0 * jnp.ones_like(_x0.abs_relaxations),
            )
        )[0]

        ## define bounds
        lower_bounds = []
        upper_bounds = []

        # (1) initial state bounds
        lb, ub = ProblemPolicyDeviation._bounds_initial_state(_x0)
        lower_bounds.append(lb.ravel())
        upper_bounds.append(ub.ravel())

        # (2) dynamics bounds
        lb, ub = ProblemPolicyDeviation._bounds_dynamics(_x0)
        lower_bounds.append(lb.ravel())
        upper_bounds.append(ub.ravel())

        # (3) policy bounds
        lb, ub = ProblemPolicyDeviation._bounds_policy(_x0)
        lower_bounds.append(lb.ravel())
        upper_bounds.append(ub.ravel())

        # (4) state-action bounds
        lb, ub = ProblemPolicyDeviation._bounds_state_action(_x0, constraint)
        lower_bounds.append(lb.ravel())
        upper_bounds.append(ub.ravel())

        # (5) relaxation bounds
        lb, ub = ProblemPolicyDeviation._bounds_relaxation(_x0)
        lower_bounds.append(lb.ravel())
        upper_bounds.append(ub.ravel())

        # print("lower_bounds", [arr.shape for arr in lower_bounds])
        # print("upper_bounds", [arr.shape for arr in upper_bounds])

        return ProblemPolicyDeviation(
            # base
            cost_linear=cost_linear,
            lower_bound=jnp.concatenate(lower_bounds),
            upper_bound=jnp.concatenate(upper_bounds),
            # env
            _system=system,
            _policy=policy,
            _constraint=constraint,
            # specifics
            _initial_state=states[0],
            _x0=_x0,
        )

    def copy_with(
        self,
        initial_state: jax.Array,
    ) -> "ProblemPolicyDeviation":
        actions = jnp.empty_like(self._x0.actions)
        states = jnp.empty_like(self._x0.states)
        relaxations = jnp.empty_like(self._x0.relaxations)

        # fill in the first state
        states = states.at[0].set(initial_state)

        # propagate actions, states and relaxations
        horizon = states.shape[0]
        for i in range(0, horizon):
            state = states[i]
            action = self._policy.get_action(state)
            next_state = self._system.step(state, action)

            actions = actions.at[i].set(action)
            if i < horizon - 1:
                states = states.at[i + 1].set(next_state)
            ineq_constraint = self._constraint.function(state, action)
            relaxations = relaxations.at[i].set(
                jnp.maximum(0, self._constraint.lower_bound - ineq_constraint)
                - jnp.maximum(0, ineq_constraint - self._constraint.upper_bound)
            )

        _x0 = ProblemPolicyDeviation.OptimizationVariable(
            actions=actions,
            policy_actions=actions,
            states=states,
            relaxations=relaxations,
            abs_relaxations=jnp.abs(relaxations),
        )

        return ProblemPolicyDeviation(
            # base
            cost_linear=self.cost_linear,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            # env
            _system=self._system,
            _policy=self._policy,
            _constraint=self._constraint,
            # specifics
            _initial_state=initial_state,
            _x0=_x0,
        )

    def _vector_to_optimization_variable(self, x: jax.Array) -> OptimizationVariable:
        _, unravel_fn = flatten_util.ravel_pytree(self._x0)
        return unravel_fn(x)

    ### QUADRATIC COST ###
    def cost_qvp(self, x: jax.Array) -> jax.Array:
        # unravel vector into struct
        optvar = self._vector_to_optimization_variable(x)

        # build Q @ x
        qvp = ProblemPolicyDeviation.OptimizationVariable(
            actions=optvar.actions - optvar.policy_actions,
            policy_actions=optvar.policy_actions - optvar.actions,
            states=jnp.zeros_like(optvar.states),
            relaxations=jnp.zeros_like(optvar.relaxations),
            abs_relaxations=jnp.zeros_like(optvar.abs_relaxations),
        )

        # flatten struct into vector
        return flatten_util.ravel_pytree(qvp)[0]

    ### NONLINEAR CONSTRAINTS ###
    def constraints(self, x: jax.Array) -> jax.Array:
        # unravel vector into struct
        optvar = self._vector_to_optimization_variable(x)

        # collect constraint function evaluations
        constraints = []
        # (1) initial state constraints
        constraints.append(self._constrains_initial_state(optvar))
        # (2) dynamics constraints
        constraints.append(self._constraints_dynamics(optvar))
        # (3) policy constraints
        constraints.append(self._constraints_policy(optvar))
        # (4) state-action constraints
        constraints.append(self._constraints_state_action(optvar))
        # (5) relaxation constraints
        constraints.append(self._constraints_relaxation(optvar))

        # print("constraints", [arr.shape for arr in constraints])

        return jnp.concatenate(constraints)

    # (1) Initial state constraints
    def _constrains_initial_state(self, optvar: OptimizationVariable) -> jax.Array:
        return optvar.states[0] - self._initial_state

    @staticmethod
    def _bounds_initial_state(optvar: OptimizationVariable) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros_like(optvar.states[0]),
            jnp.zeros_like(optvar.states[0]),
        )

    # (2) Dynamics constraints
    def _constraints_dynamics(self, optvar: OptimizationVariable) -> jax.Array:
        return jax.vmap(
            lambda state, action, next_state: next_state - self._system.step(state, action)
        )(optvar.states[:-1], optvar.actions[:-1], optvar.states[1:]).ravel()

    @staticmethod
    def _bounds_dynamics(optvar: OptimizationVariable) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros(optvar.states.size - optvar.states[0].size),
            jnp.zeros(optvar.states.size - optvar.states[0].size),
        )

    # (3) Policy constraints
    def _constraints_policy(self, optvar: OptimizationVariable) -> jax.Array:
        return jax.vmap(
            lambda state, policy_action: policy_action - self._policy.get_action(state)
        )(optvar.states, optvar.policy_actions).ravel()

    @staticmethod
    def _bounds_policy(optvar: OptimizationVariable) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros(optvar.actions.size),
            jnp.zeros(optvar.actions.size),
        )

    # (4) State action constraints
    def _constraints_state_action(self, optvar: OptimizationVariable) -> jax.Array:
        return jax.vmap(
            lambda state, action, relaxation: self._constraint.function(state, action) + relaxation
        )(optvar.states, optvar.actions, optvar.relaxations).ravel()

    @staticmethod
    def _bounds_state_action(
        optvar: OptimizationVariable, constraint: StateActionConstraint
    ) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.tile(constraint.lower_bound, optvar.relaxations.shape[0]),
            jnp.tile(constraint.upper_bound, optvar.relaxations.shape[0]),
        )

    # (5) Relaxation absolute value constraints
    def _constraints_relaxation(self, optvar: OptimizationVariable) -> jax.Array:
        return jnp.concatenate(
            (
                optvar.relaxations.ravel() - optvar.abs_relaxations.ravel(),
                optvar.relaxations.ravel() + optvar.abs_relaxations.ravel(),
                optvar.abs_relaxations.ravel(),
            )
        )

    @staticmethod
    def _bounds_relaxation(optvar: OptimizationVariable) -> tuple[jax.Array, jax.Array]:
        return (
            # lower
            jnp.concatenate(
                (
                    -jnp.inf * jnp.ones(optvar.relaxations.size),
                    jnp.zeros(2 * optvar.relaxations.size),
                )
            ),
            # upper
            jnp.concatenate(
                (
                    jnp.zeros(optvar.relaxations.size),
                    jnp.inf * jnp.ones(2 * optvar.relaxations.size),
                )
            ),
        )

    ### OUTPUT ###
    def pretty_print(self, x: jax.Array):
        # unravel vector into struct
        optvar = self._vector_to_optimization_variable(x)

        print("actions", optvar.actions)
        print("policy_actions", optvar.policy_actions)
        print("states", optvar.states)
        print("relaxations", optvar.relaxations)
        print("abs_relaxations", optvar.abs_relaxations)

    def debug_optvar(self, x: jax.Array):
        return self._vector_to_optimization_variable(x)

    def get_action(self, x: jax.Array):
        optvar = self._vector_to_optimization_variable(x)
        return optvar.actions[0]
