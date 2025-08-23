import jax
import jax.numpy as jnp
from flax import struct

from hcnn.optimization_layer.projection_problem import (
    ProjectionPhi,
    ProjectionProblem,
)
from hcnn.optimization_layer.rollout import Rollout
from hcnn.solver import SQPInfo, SQPParameters, sqp_solve
from hcnn.system_constraints import SystemConstraintsBase
from hcnn.system_dynamics import SystemDynamicsBase


@struct.dataclass
class ProjectionInfo(SQPInfo):
    state_diff: jax.Array
    action_diff: jax.Array


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
        sqp_parameters: SQPParameters,
    ) -> "ProjectionLayer":
        return cls(
            projection_problem=ProjectionProblem.factory(
                sys_dynamics=sys_dynamics,
                sys_constraints=sys_constraints,
                horizon=horizon,
            ),
            sqp_parameters=sqp_parameters,
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
            # Construct and solve SQP Problem
            phi = ProjectionPhi(
                states_ref=rollout.states,
                actions_ref=rollout.actions,
                initial_state=initial_state,
            )
            sqp_problem = self.projection_problem.replace(phi=phi)
            x_star, sqp_info = sqp_solve(sqp_problem, self.sqp_parameters)

            # Extract optimal variables
            optvar = self.projection_problem.x_to_optvar(x_star)
            rollout_proj = Rollout(
                states=optvar.states,
                actions=optvar.actions,
            )
            state_diff = jnp.mean(
                jnp.abs(
                    rollout.states - jax.lax.stop_gradient(rollout_proj.states)
                ),
            )
            action_diff = jnp.mean(
                jnp.abs(
                    rollout.actions
                    - jax.lax.stop_gradient(rollout_proj.actions)
                ),
            )
            proj_info = ProjectionInfo(
                primal_residual=sqp_info.primal_residual,
                dual_residual=sqp_info.dual_residual,
                iterations_lin=sqp_info.iterations_lin,
                iterations_nonlin=sqp_info.iterations_nonlin,
                rho_base_mult=sqp_info.rho_base_mult,
                converged_flag=sqp_info.converged_flag,
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
