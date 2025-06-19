import jax
import jax.numpy as jnp
from flax import nnx, struct
from jax import flatten_util

from hcnn.policy import PolicyBase
from hcnn.projection import ProjectionBase, ProjectionPhi, Rollout
from hcnn.system_constraints import SystemConstraintsBase
from hcnn.system_dynamics import SystemDynamicsBase


@struct.dataclass
class ProjectionNoSubstitutionNoTerminal(ProjectionBase):
    @classmethod
    def factory(
        cls,
        policy: PolicyBase,
        sys_dynamics: SystemDynamicsBase,
        sys_constraints: SystemConstraintsBase,
        horizon: int,
        relaxation_penalty: float,
    ):
        # OptimizationProblem
        cost_quad_diag, x_to_rollout = flatten_util.ravel_pytree(
            Rollout(
                actions=jnp.zeros((horizon, sys_dynamics.action_dim)),
                states=jnp.zeros((horizon, sys_dynamics.state_dim)),
                interventions=2 * jnp.ones((horizon, sys_dynamics.action_dim)),
                relaxations_plus=jnp.zeros(horizon),
                relaxations_minus=jnp.zeros(horizon),
            )
        )
        cost_linear, _ = flatten_util.ravel_pytree(
            Rollout(
                actions=jnp.zeros((horizon, sys_dynamics.action_dim)),
                states=jnp.zeros((horizon, sys_dynamics.state_dim)),
                interventions=jnp.zeros((horizon, sys_dynamics.action_dim)),
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
        lower_bound_list, upper_bound_list = zip(
            *bound_tuple_list, strict=False
        )
        lower_bound = jnp.concatenate(lower_bound_list)
        upper_bound = jnp.concatenate(upper_bound_list)

        problem = ProjectionNoSubstitutionNoTerminal(
            cost_quad_diag=cost_quad_diag,
            cost_linear=cost_linear,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        phi = ProjectionPhi(
            initial_state=jnp.zeros((sys_dynamics.state_dim,)),
            split_policy=nnx.split(policy),
            split_dynamics=nnx.split(sys_dynamics),
            split_constraints=nnx.split(sys_constraints),
            jax_key=jax.random.key(0),
            horizon=horizon,
            x_to_rollout=x_to_rollout,
        )

        return (problem, phi)

    @staticmethod
    @jax.jit
    def get_x0(phi: ProjectionPhi) -> jax.Array:
        policy = nnx.merge(*phi.split_policy)
        sys_dynamics = nnx.merge(*phi.split_dynamics)
        sys_constraints = nnx.merge(*phi.split_constraints)

        def next_row(state, i):
            action = policy(state, jax_key=phi.jax_key)
            intervention = jnp.zeros_like(action)
            next_state = sys_dynamics.next_state(state, action)
            constraint = sys_constraints(state, action)
            relaxation_plus = jnp.max(
                jnp.maximum(0.0, sys_constraints.lower_bound - constraint)
            )
            relaxations_minus = jnp.max(
                jnp.max(
                    jnp.maximum(0.0, constraint - sys_constraints.upper_bound)
                ),
            )
            out = (
                action,
                state,
                intervention,
                relaxation_plus,
                relaxations_minus,
            )
            return next_state, out

        (_, rollout_tuple) = jax.lax.scan(
            f=next_row,
            init=phi.initial_state,
            xs=jnp.arange(phi.horizon),
        )

        optvar = Rollout(
            actions=rollout_tuple[0],
            states=rollout_tuple[1],
            interventions=rollout_tuple[2],
            relaxations_plus=rollout_tuple[3],
            relaxations_minus=rollout_tuple[4],
        )

        return flatten_util.ravel_pytree(optvar)[0]

    @staticmethod
    @jax.jit
    def constraint_fn(x: jax.Array, phi: ProjectionPhi) -> jax.Array:
        rollout = phi.x_to_rollout(x)
        policy = nnx.merge(*phi.split_policy)
        sys_dynamics = nnx.merge(*phi.split_dynamics)
        sys_constraints = nnx.merge(*phi.split_constraints)

        constraint_fun_list = [
            ProjectionNoSubstitutionNoTerminal._constraints_initial_state(
                rollout, phi.initial_state
            ),
            ProjectionNoSubstitutionNoTerminal._constraints_sys_dynamics(
                rollout, sys_dynamics
            ),
            ProjectionNoSubstitutionNoTerminal._constraints_policy(
                rollout, policy, phi.jax_key
            ),
            ProjectionNoSubstitutionNoTerminal._constraints_sys_constraints(
                rollout, sys_constraints
            ),
            ProjectionNoSubstitutionNoTerminal._constraints_non_neg_relax(
                rollout
            ),
        ]
        return jnp.concatenate(constraint_fun_list)

    # (1) initial state
    @staticmethod
    def _constraints_initial_state(
        rollout: Rollout, initial_state: jax.Array
    ) -> jax.Array:
        return rollout.states[0] - initial_state

    @staticmethod
    def _bounds_initial_state(sys_dynamics) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros(sys_dynamics.state_dim),
            jnp.zeros(sys_dynamics.state_dim),
        )

    # (2) system dynamics
    @staticmethod
    def _constraints_sys_dynamics(rollout: Rollout, sys_dynamics) -> jax.Array:
        def next_state_constr(data_slice):
            state_i, state_i_plus_1, action_i = data_slice
            pred_next_state = sys_dynamics.next_state(state_i, action_i)
            constraint = state_i_plus_1 - pred_next_state
            return constraint

        states_i = rollout.states[:-1]
        states_i_plus_1 = rollout.states[1:]
        actions_i = rollout.actions[:-1]

        constraints = jax.lax.map(
            next_state_constr,
            (states_i, states_i_plus_1, actions_i),
        )
        return constraints.ravel()

    @staticmethod
    def _bounds_sys_dynamics(
        horizon, sys_dynamics
    ) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros((horizon - 1) * sys_dynamics.state_dim),
            jnp.zeros((horizon - 1) * sys_dynamics.state_dim),
        )

    # (3) policy
    @staticmethod
    def _constraints_policy(rollout: Rollout, policy, jax_key) -> jax.Array:
        def policy_constraints(data_slice):
            state, action, intervention = data_slice
            policy_action = policy(state, jax_key=jax_key)
            constraint = action - policy_action - intervention
            return constraint

        # Prepare data slices for map
        data_slices = (
            rollout.states,
            rollout.actions,
            rollout.interventions,
        )

        constraints = jax.lax.map(
            policy_constraints,
            data_slices,
        )

        # constraints = rollout.actions - policy(rollout.states, jax_key=jax_key) - rollout.interventions
        return constraints.ravel()

    @staticmethod
    def _bounds_policy(horizon, sys_dynamics) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros(sys_dynamics.action_dim * horizon),
            jnp.zeros(sys_dynamics.action_dim * horizon),
        )

    # (4) system constraints
    @staticmethod
    def _constraints_sys_constraints(
        rollout: Rollout, sys_constraints
    ) -> jax.Array:
        def sys_constraint_fn(data_slice):
            state, action, r_plus, r_minus = data_slice
            constraint = sys_constraints(state, action) + r_plus - r_minus
            return constraint

        # Prepare data slices for map
        data_slices = (
            rollout.states,
            rollout.actions,
            rollout.relaxations_plus,
            rollout.relaxations_minus,
        )

        constraints = jax.lax.map(
            sys_constraint_fn,
            data_slices,
        )

        return constraints.ravel()

    @staticmethod
    def _bounds_sys_constraints(
        horizon, sys_constraints
    ) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.tile(sys_constraints.lower_bound, horizon),
            jnp.tile(sys_constraints.upper_bound, horizon),
        )

    # (5) non-negative relaxations
    @staticmethod
    def _constraints_non_neg_relax(rollout: Rollout) -> jax.Array:
        return jnp.concatenate(
            (rollout.relaxations_plus, rollout.relaxations_minus)
        )

    @staticmethod
    def _bounds_non_neg_relax(horizon) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros(2 * horizon),
            jnp.ones(2 * horizon) * jnp.inf,
        )
