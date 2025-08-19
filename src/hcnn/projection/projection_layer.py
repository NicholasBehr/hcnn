import jax
import jax.numpy as jnp
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
class ProjectionInfo(OptimizationInfo):
    state_diff: jax.Array
    action_diff: jax.Array


@struct.dataclass
class ProjectionLayer:
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
            projection_problem=ProjectionProblem.factory(
                sys_dynamics=sys_dynamics,
                sys_constraints=sys_constraints,
                horizon=horizon,
            ),
            solver=solver,
        )

    def __call__(
        self, rollout: Rollout, initial_state: jax.Array
    ) -> tuple[Rollout, ProjectionInfo]:
        """Project a sequence of actions onto the system dynamics.
        Args:
            rollout: Sequence of actions and states,
                shape (..., horizon, state_dim) for states and
                (..., horizon, action_dim) for actions
            initial_state: Initial state, shape (..., state_dim)

        Returns:
            A tuple containing:
                - The projected rollout
                - Additional information
        """

        def sample_rollout(
            rollout: Rollout,
            initial_state: jax.Array,
        ):
            phi_d = ProjectionPhiD(
                states_ref=rollout.states,
                actions_ref=rollout.actions,
            )
            phi_nd = ProjectionPhiND(
                initial_state=initial_state,
            )
            x_star, solver_info = self.solver.solve(
                self.projection_problem, phi_d, phi_nd
            )
            optvar = self.projection_problem.x_to_optvar(x_star)
            rollout_proj = Rollout(
                states=optvar.states,
                actions=optvar.actions,
            )
            state_diff = jnp.mean(
                jnp.abs(rollout.states - rollout_proj.states),
            )
            action_diff = jnp.mean(
                jnp.abs(rollout.actions - rollout_proj.actions),
            )
            proj_info = ProjectionInfo(
                primal_residual=solver_info.primal_residual,
                dual_residual=solver_info.dual_residual,
                iterations_lin=solver_info.iterations_lin,
                iterations_nonlin=solver_info.iterations_nonlin,
                converged_flag=solver_info.converged_flag,
                state_diff=state_diff,
                action_diff=action_diff,
            )
            return rollout_proj, proj_info

        # Check if we need to vmap based on the batch dimensions
        # Single rollout: states.shape = (horizon, state_dim)
        # Batch rollout: states.shape = (batch_size, horizon, state_dim)
        if rollout.states.ndim > 2:  # batch dimension present?  # noqa: PLR2004
            return jax.vmap(
                sample_rollout,
                in_axes=(0, 0),
            )(rollout, initial_state)
        else:
            return sample_rollout(rollout, initial_state)
