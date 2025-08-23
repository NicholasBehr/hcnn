import jax
import jax.numpy as jnp
from flax import struct
from jax import flatten_util

from hcnn.solver import SQPProblemBase
from hcnn.system_constraints import SystemConstraintsBox
from hcnn.system_costs import SystemCostsQuadratic
from hcnn.system_dynamics import SystemDynamicsLinear


@struct.dataclass
class OptimizationVariable:
    states: jax.Array  # shape (horizon, state_dim)
    actions: jax.Array  # shape (horizon, action_dim)


@struct.dataclass
class MPCPhi:
    initial_state: jax.Array  # shape (state_dim,)


@struct.dataclass
class MPCProblem(SQPProblemBase[MPCPhi]):
    sys_dynamics: SystemDynamicsLinear
    sys_constraints: SystemConstraintsBox
    sys_costs: SystemCostsQuadratic
    horizon: int = struct.field(pytree_node=False)

    @classmethod
    def factory(
        cls,
        sys_dynamics: SystemDynamicsLinear,
        sys_constraints: SystemConstraintsBox,
        sys_costs: SystemCostsQuadratic,
        horizon: int,
    ):
        state_dim = sys_constraints.state_dim
        action_dim = sys_constraints.action_dim

        # Build quadratic cost diagonal from system costs
        # For MPC: min sum_t ||x_t - x_ref_t||_Q^2 + ||u_t - u_ref_t||_R^2
        state_cost_diag = jnp.diag(sys_costs.state_cost_matrix)
        action_cost_diag = jnp.diag(sys_costs.action_cost_matrix)

        # Repeat for each time step
        state_costs_tiled = jnp.tile(state_cost_diag, horizon)
        action_costs_tiled = jnp.tile(action_cost_diag, horizon)

        cost_quad_diag = flatten_util.ravel_pytree(
            OptimizationVariable(
                states=state_costs_tiled.reshape(horizon, state_dim),
                actions=action_costs_tiled.reshape(horizon, action_dim),
            )
        )[0]

        # Linear cost includes reference tracking terms and relaxation penalty
        # For quadratic cost (x-r)^T Q (x-r) = x^T Q x - 2 r^T Q x + r^T Q r
        # The linear term is -2 Q r, where r is the constant reference

        # Create reference trajectories (constant reference repeated)
        states_ref_flat = jnp.tile(sys_costs.reference_state, horizon)
        actions_ref_flat = jnp.tile(sys_costs.reference_action, horizon)

        # Compute linear cost: -2 * diag(Q) * r
        state_linear = -2 * state_costs_tiled * states_ref_flat
        action_linear = -2 * action_costs_tiled * actions_ref_flat

        cost_linear = flatten_util.ravel_pytree(
            OptimizationVariable(
                states=state_linear.reshape(horizon, state_dim),
                actions=action_linear.reshape(horizon, action_dim),
            )
        )[0]

        phi = MPCPhi(
            initial_state=jnp.zeros((state_dim,)),
        )

        bound_tuple_list = [
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
            sys_costs=sys_costs,
            horizon=horizon,
        )

    ## SYSTEM DYNAMICS CONSTRAINTS ###
    @staticmethod
    def constraints_sys_dynamics(
        optvar: OptimizationVariable, sys_dynamics: SystemDynamicsLinear
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
        optvar: OptimizationVariable, sys_constraints: SystemConstraintsBox
    ) -> jax.Array:
        return sys_constraints(optvar.states, optvar.actions).ravel()

    @staticmethod
    def bounds_sys_constraints(
        horizon: int, sys_constraints: SystemConstraintsBox
    ) -> tuple[jax.Array, jax.Array]:
        # Tile bounds for each time step to match raveled constraint structure
        lower_bounds = jnp.tile(sys_constraints.lower_bound, horizon)
        upper_bounds = jnp.tile(sys_constraints.upper_bound, horizon)
        return (lower_bounds, upper_bounds)

    ## INITIAL STATE CONSTRAINTS ###
    @staticmethod
    def constraints_initial_state(
        optvar: OptimizationVariable, phi: MPCPhi
    ) -> jax.Array:
        return optvar.states[0] - phi.initial_state

    @staticmethod
    def bounds_initial_state(state_dim: int) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros(state_dim),
            jnp.zeros(state_dim),
        )

    def get_x0(self, phi: MPCPhi) -> jax.Array:
        # Create reference trajectory from constant references
        states_ref = jnp.tile(
            self.sys_costs.reference_state[None, :], (self.horizon, 1)
        )
        actions_ref = jnp.tile(
            self.sys_costs.reference_action[None, :], (self.horizon, 1)
        )

        optvar = OptimizationVariable(
            states=states_ref,
            actions=actions_ref,
        )
        return flatten_util.ravel_pytree(optvar)[0]

    def x_to_optvar(self, x: jax.Array) -> OptimizationVariable:
        state_dim = self.sys_constraints.state_dim
        action_dim = self.sys_constraints.action_dim
        state_template = jnp.empty((self.horizon, state_dim))
        action_template = jnp.empty((self.horizon, action_dim))

        optvar_template = OptimizationVariable(
            states=state_template,
            actions=action_template,
        )
        _, unravel_pytree = flatten_util.ravel_pytree(optvar_template)
        return unravel_pytree(x)

    def constraint_fn_phi(self, x, phi):
        optvar = self.x_to_optvar(x)
        constraints_list = [
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
