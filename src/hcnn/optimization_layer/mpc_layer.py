import jax
from flax import struct

from hcnn.optimization_layer.mpc_problem import MPCPhi, MPCProblem
from hcnn.optimization_layer.rollout import Rollout
from hcnn.solver import SQPInfo, SQPParameters, sqp_solve
from hcnn.system_constraints import SystemConstraintsBox
from hcnn.system_costs import SystemCostsQuadratic
from hcnn.system_dynamics import SystemDynamicsLinear


@struct.dataclass
class MPCLayer:
    mpc_problem: MPCProblem
    sqp_parameters: SQPParameters

    @classmethod
    def factory(
        cls,
        sys_dynamics: SystemDynamicsLinear,
        sys_constraints: SystemConstraintsBox,
        sys_costs: SystemCostsQuadratic,
        horizon: int,
        sqp_parameters: SQPParameters,
    ) -> "MPCLayer":
        return cls(
            mpc_problem=MPCProblem.factory(
                sys_dynamics=sys_dynamics,
                sys_constraints=sys_constraints,
                sys_costs=sys_costs,
                horizon=horizon,
            ),
            sqp_parameters=sqp_parameters,
        )

    def __call__(self, initial_state: jax.Array) -> tuple[Rollout, SQPInfo]:
        """Solve MPC problem to get optimal control trajectory.

        Args:
            initial_state: Initial state, shape (..., state_dim)

        Returns:
            A tuple containing:
                - The optimal rollout (states, actions, relaxation)
                - Additional information about the optimization
        """

        def solve_mpc(initial_state: jax.Array):
            # Construct and solve SQP Problem
            phi = MPCPhi(
                initial_state=initial_state,
            )
            sqp_problem = self.mpc_problem.replace(phi=phi)
            x_star, sqp_info = sqp_solve(sqp_problem, self.sqp_parameters)

            # Extract optimal variables
            optvar = self.mpc_problem.x_to_optvar(x_star)

            # Create the rollout
            rollout = Rollout(
                states=optvar.states,
                actions=optvar.actions,
            )

            return rollout, sqp_info

        # Check if we need to vmap based on the batch dimensions
        # Single initial state: initial_state.shape = (state_dim,)
        # Batch initial states: initial_state.shape = (batch_size, state_dim)
        if initial_state.ndim > 1:  # batch dimension present?
            return jax.vmap(solve_mpc)(initial_state)
        else:
            return solve_mpc(initial_state)
