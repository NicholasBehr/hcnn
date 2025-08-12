import jax
from flax import struct

from hcnn.optimization.optimization_base import (
    OptimizationInfo,
    OptimizationSolverBase,
)
from hcnn.system_constraints import SystemConstraintsBase
from hcnn.system_dynamics import SystemDynamicsBase

from .projection_problem import (
    ProjectionPhiD,
    ProjectionPhiND,
    ProjectionProblem,
)
from .rollout import Rollout


@struct.dataclass
class ProjectionLayer:
    sys_dynamics: SystemDynamicsBase
    horizon: int
    projection_problem: ProjectionProblem
    solver: OptimizationSolverBase

    @classmethod
    def factory(
        cls,
        sys_dynamics: SystemDynamicsBase,
        sys_constraints: SystemConstraintsBase,
        horizon: int,
        solver: OptimizationSolverBase,
    ) -> "ProjectionLayer":
        return cls(
            sys_dynamics=sys_dynamics,
            horizon=horizon,
            projection_problem=ProjectionProblem.factory(
                sys_constraints, horizon
            ),
            solver=solver,
        )

    def __call__(
        self, rollout: Rollout, initial_state: jax.Array
    ) -> tuple[Rollout, Rollout, OptimizationInfo]:
        """Project a sequence of actions onto the system dynamics.
        Args:
            rollout: Sequence of actions and states,
                shape (..., horizon, state_dim) for states and
                (..., horizon, action_dim) for actions
            initial_state: Initial state, shape (..., state_dim)

        Returns:
            A tuple containing:
                - The projected rollout
                - The difference between the original and projected rollout
                - Additional information
        """

        def sample_rollout(
            rollout: Rollout,
            initial_state: jax.Array,
        ):
            phi_d = ProjectionPhiD(
                rollout_ref=rollout,
            )
            phi_nd = ProjectionPhiND(
                sys_dynamics=self.sys_dynamics,
                initial_state=initial_state,
            )
            x_star, info = self.solver.solve(
                self.projection_problem, phi_d, phi_nd
            )
            optvar = self.projection_problem.x_to_optvar(x_star)
            return optvar.rollout, optvar.rollout_diff, info

        # Check if we need to vmap based on the batch dimensions
        # Single rollout: states.shape = (horizon, state_dim)
        # Batch rollout: states.shape = (batch_size, horizon, state_dim)
        if rollout.states.ndim > 2:  # batch dimension present
            return jax.vmap(sample_rollout)(rollout, initial_state)
        else:
            return sample_rollout(rollout, initial_state)
