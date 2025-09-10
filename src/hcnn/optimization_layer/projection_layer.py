import jax
from flax import struct

from hcnn.optimization_layer.projection_problem import (
    ProjectionOptimizationVariable,
    ProjectionPhi,
    ProjectionProblem,
)
from hcnn.optimization_layer.rollout import Rollout
from hcnn.solver import SQPInfo, SQPParameters, sqp_solve
from hcnn.system_constraints import SystemConstraintsBase
from hcnn.system_dynamics import SystemDynamicsBase


@struct.dataclass
class ProjectionLayer:
    projection_problem: ProjectionProblem
    sqp_parameters: SQPParameters

    @classmethod
    def factory(
        cls,
        sys_dynamics: SystemDynamicsBase,
        sys_constraints: SystemConstraintsBase,
        horizon: int,
        projection_relaxation_penalty: float,
        sqp_parameters: SQPParameters,
    ) -> "ProjectionLayer":
        return cls(
            projection_problem=ProjectionProblem.factory(
                sys_dynamics=sys_dynamics,
                sys_constraints=sys_constraints,
                horizon=horizon,
                projection_relaxation_penalty=projection_relaxation_penalty,
            ),
            sqp_parameters=sqp_parameters,
        )

    def __call__(
        self, rollout: Rollout, initial_state: jax.Array
    ) -> tuple[ProjectionOptimizationVariable, SQPInfo]:
        """Project a sequence of actions onto the system dynamics.
        Args:
            rollout: Sequence of actions and states,
                shape (..., horizon, state_dim) for states and
                (..., horizon, action_dim) for actions
            initial_state: Initial state, shape (..., state_dim)
        """

        def sample_rollout(
            rollout: Rollout,
            initial_state: jax.Array,
        ):
            # Construct and solve SQP Problem
            phi = ProjectionPhi(
                states_ref=rollout.states,
                actions_ref=rollout.actions,
                initial_state=initial_state,
            )
            sqp_problem = self.projection_problem.replace(phi=phi)
            x_star, sqp_info = sqp_solve(sqp_problem, self.sqp_parameters)
            proj_optvar = self.projection_problem.x_to_optvar(x_star)
            proj_optvar.replace(
                # replace so we get allignment gradients,
                # even if proj doesnt converge
                states_diff=jax.lax.stop_gradient(proj_optvar.states)
                - rollout.states,
                actions_diff=jax.lax.stop_gradient(proj_optvar.actions)
                - rollout.actions,
            )

            return proj_optvar, sqp_info

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
