from typing import Optional

import jax
import jax.numpy as jnp
from flax import struct

from hcnn.system_costs.system_costs_base import SystemCostsBase
from hcnn.system_dynamics.system_dynamics_double_pendulum import (
    SystemDynamicsDoublePendulum,
)


@struct.dataclass
class SystemCostsDoublePendulum(SystemCostsBase):
    """
    Cost function for double pendulum based on quadratic difference from
    reference pose.

    Computes the cost as ||fk(q) - fk_ref||²_W where:
    - fk(q) is the forward kinematics of the current state
    - fk_ref is the reference forward kinematics pose computed from reference
      generalized coordinates
    - W is a weight matrix for the position coordinates

    The forward kinematics returns [x1, y1, x2, y2] where:
    - (x1, y1) is the position of the end of link 1
    - (x2, y2) is the position of the end of link 2
    """

    dynamics: SystemDynamicsDoublePendulum
    reference_pose: jax.Array  # Shape (4,) - [x1_ref, y1_ref, x2_ref, y2_ref]
    reference_state: jax.Array  # Shape (4,) - [θ₁_ref, θ₂_ref, θ̇₁_ref, θ̇₂_ref]
    position_weight_matrix: jax.Array  # Shape (4, 4) - weight matrix
    action_cost_matrix: jax.Array  # Shape (action_dim, action_dim)

    @classmethod
    def factory(
        cls,
        dynamics: SystemDynamicsDoublePendulum,
        reference_state: jax.Array,
        position_weights: Optional[jax.Array] = None,
        action_weights: Optional[jax.Array] = None,
    ) -> "SystemCostsDoublePendulum":
        """
        Factory method to create a forward kinematics cost for double pendulum.

        Args:
            dynamics: SystemDynamicsDoublePendulum instance for computing
                forward kinematics
            reference_state: Reference generalized coordinates
                [θ₁_ref, θ₂_ref, θ̇₁_ref, θ̇₂_ref]
            position_weights: Weights for position errors. If None, uses
                identity. Can be a scalar, 1D array (diagonal), or 2D matrix.
            action_weights: Weights for action costs. If None, uses identity.
                Can be a scalar, 1D array (diagonal), or 2D matrix.

        Returns:
            SystemCostsDoublePendulumForwardKinematics instance
        """
        # Get JAX's default float type to respect 32/64 bit config
        float_dtype = jnp.ones(1).dtype

        # Process reference state
        reference_state = jnp.asarray(reference_state, dtype=float_dtype)
        error_msg = "Reference state must be shape (4,) for [θ₁, θ₂, θ̇₁, θ̇₂]"
        assert reference_state.shape == (4,), error_msg

        # Compute reference pose from reference state
        reference_pose = dynamics.forward_kinematics(reference_state)

        # Process position weights
        if position_weights is None:
            position_weight_matrix = jnp.eye(4, dtype=float_dtype)
        else:
            position_weights = jnp.asarray(position_weights, dtype=float_dtype)
            if position_weights.ndim == 0:  # scalar
                eye_matrix = jnp.eye(4, dtype=float_dtype)
                position_weight_matrix = position_weights * eye_matrix
            elif position_weights.ndim == 1:  # diagonal
                error_msg = "1D position weights must have shape (4,)"
                assert position_weights.shape == (4,), error_msg
                position_weight_matrix = jnp.diag(position_weights)
            elif position_weights.ndim == 2:  # full matrix  # noqa: PLR2004
                shape = (4, 4)
                error_msg = "2D position weights must have shape (4, 4)"
                assert position_weights.shape == shape, error_msg
                position_weight_matrix = position_weights
            else:
                raise ValueError("Position weights must be 0D, 1D, or 2D array")

        # Process action weights
        action_dim = dynamics.action_dim
        if action_weights is None:
            action_cost_matrix = jnp.eye(action_dim, dtype=float_dtype)
        else:
            action_weights = jnp.asarray(action_weights, dtype=float_dtype)
            if action_weights.ndim == 0:  # scalar
                eye_matrix = jnp.eye(action_dim, dtype=float_dtype)
                action_cost_matrix = action_weights * eye_matrix
            elif action_weights.ndim == 1:  # diagonal
                error_msg = f"1D action weights must have shape ({action_dim},)"
                assert action_weights.shape == (action_dim,), error_msg
                action_cost_matrix = jnp.diag(action_weights)
            elif action_weights.ndim == 2:  # full matrix  # noqa: PLR2004
                shape = (action_dim, action_dim)
                error_msg = (
                    f"2D action weights must have shape "
                    f"({action_dim}, {action_dim})"
                )
                assert action_weights.shape == shape, error_msg
                action_cost_matrix = action_weights
            else:
                raise ValueError("Action weights must be 0D, 1D, or 2D array")

        return cls(
            state_dim=dynamics.state_dim,
            action_dim=dynamics.action_dim,
            dynamics=dynamics,
            reference_pose=reference_pose,
            reference_state=reference_state,
            position_weight_matrix=position_weight_matrix,
            action_cost_matrix=action_cost_matrix,
        )

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """
        Compute the cost as quadratic difference from reference pose plus
        action cost.

        Args:
            state: State array of shape (..., 4) containing [θ₁, θ₂, θ̇₁, θ̇₂]
            action: Action array of shape (..., 1) containing [τ₁]

        Returns:
            Cost value of shape (...,)
        """
        # Compute current forward kinematics
        current_pose = self.dynamics.forward_kinematics(state)  # (..., 4)

        # Compute pose error
        pose_error = current_pose - self.reference_pose  # (..., 4)

        # Compute quadratic cost for pose error
        # cost = e^T W e where e is the pose error and W is the weight matrix
        position_cost = jnp.sum(
            (pose_error @ self.position_weight_matrix) * pose_error,
            axis=-1,
        )

        # Compute action cost
        action_cost = jnp.sum(
            (action @ self.action_cost_matrix) * action,
            axis=-1,
        )

        return position_cost + action_cost
