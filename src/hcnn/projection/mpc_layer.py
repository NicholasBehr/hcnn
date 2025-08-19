import jax
from flax import struct

from hcnn.optimization.optimization_base import (
    OptimizationInfo,
    OptimizationSolverBase,
)
from hcnn.system_constraints import SystemConstraintsBox
from hcnn.system_costs import SystemCostsQuadratic
from hcnn.system_dynamics import SystemDynamicsLinear

from .mpc_problem import (
    MPCPhiD,
    MPCPhiND,
    MPCProblem,
)
from .rollout import Rollout


@struct.dataclass
class MPCLayer:
    mpc_problem: MPCProblem
    solver: OptimizationSolverBase

    @classmethod
    def factory(
        cls,
        sys_dynamics: SystemDynamicsLinear,
        sys_constraints: SystemConstraintsBox,
        sys_costs: SystemCostsQuadratic,
        horizon: int,
        solver,
    ) -> "MPCLayer":
        return cls(
            mpc_problem=MPCProblem.factory(
                sys_dynamics=sys_dynamics,
                sys_constraints=sys_constraints,
                sys_costs=sys_costs,
                horizon=horizon,
            ),
            solver=solver,
        )

    def __call__(
        self, initial_state: jax.Array
    ) -> tuple[Rollout, OptimizationInfo]:
        """Solve MPC problem to get optimal control trajectory.

        Args:
            initial_state: Initial state, shape (..., state_dim)

        Returns:
            A tuple containing:
                - The optimal rollout (states, actions, relaxation)
                - Additional information about the optimization
        """

        def solve_mpc(initial_state: jax.Array):
            phi_d = MPCPhiD()  # Empty since reference comes from sys_costs
            phi_nd = MPCPhiND(
                initial_state=initial_state,
            )

            x_star, solver_info = self.solver.solve(
                self.mpc_problem, phi_d, phi_nd
            )
            optvar = self.mpc_problem.x_to_optvar(x_star)

            # Create the rollout
            rollout = Rollout(
                states=optvar.states,
                actions=optvar.actions,
            )

            return rollout, solver_info

        # Check if we need to vmap based on the batch dimensions
        # Single initial state: initial_state.shape = (state_dim,)
        # Batch initial states: initial_state.shape = (batch_size, state_dim)
        if initial_state.ndim > 1:  # batch dimension present?
            return jax.vmap(solve_mpc)(initial_state)
        else:
            return solve_mpc(initial_state)
