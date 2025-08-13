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
    relax_diff: jax.Array
    relax_mean: jax.Array


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
        relaxation_penalty: float,
        solver: OptimizationSolverBase,
    ) -> "ProjectionLayer":
        return cls(
            sys_dynamics=sys_dynamics,
            horizon=horizon,
            projection_problem=ProjectionProblem.factory(
                sys_constraints=sys_constraints,
                horizon=horizon,
                relaxation_penalty=relaxation_penalty,
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
            states_relax_pos = +jnp.clip(rollout.states_relax, 0, None)
            states_relax_neg = -jnp.clip(rollout.states_relax, None, 0)
            phi_d = ProjectionPhiD(
                states_ref=rollout.states,
                actions_ref=rollout.actions,
            )
            phi_nd = ProjectionPhiND(
                sys_dynamics=self.sys_dynamics,
                initial_state=initial_state,
                states_relax_pos=states_relax_pos,
                states_relax_neg=states_relax_neg,
            )
            x_star, solver_info = self.solver.solve(
                self.projection_problem, phi_d, phi_nd
            )
            optvar = self.projection_problem.x_to_optvar(x_star)
            rollout_proj = Rollout(
                states=optvar.states,
                actions=optvar.actions,
                states_relax=optvar.states_relax_pos - optvar.states_relax_neg,
            )
            state_diff = jnp.mean(
                jnp.abs(rollout.states - rollout_proj.states),
            )
            action_diff = jnp.mean(
                jnp.abs(rollout.actions - rollout_proj.actions),
            )
            relax_diff = jnp.mean(
                jnp.abs(rollout.states_relax - rollout_proj.states_relax),
            )
            relax_mean = jnp.mean(jnp.abs(rollout_proj.states_relax))
            proj_info = ProjectionInfo(
                primal_residual=solver_info.primal_residual,
                dual_residual=solver_info.dual_residual,
                iterations_lin=solver_info.iterations_lin,
                iterations_nonlin=solver_info.iterations_nonlin,
                converged_flag=solver_info.converged_flag,
                state_diff=state_diff,
                action_diff=action_diff,
                relax_diff=relax_diff,
                relax_mean=relax_mean,
            )
            return rollout, proj_info

        # Check if we need to vmap based on the batch dimensions
        # Single rollout: states.shape = (horizon, state_dim)
        # Batch rollout: states.shape = (batch_size, horizon, state_dim)
        if rollout.states.ndim > 2:  # batch dimension present?  # noqa: PLR2004
            return jax.vmap(sample_rollout)(rollout, initial_state)
        else:
            return sample_rollout(rollout, initial_state)
