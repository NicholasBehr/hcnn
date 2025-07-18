import jax
import jax.numpy as jnp
from flax import struct
from mujoco import mjx

from hcnn.system_dynamics.system_dynamics_base import SystemDynamicsBase


@struct.dataclass
class SystemDynamicsMuJoCo(SystemDynamicsBase):
    mjx_model: mjx.Model
    sim_per_control_step: int = struct.field(pytree_node=False)

    @classmethod
    def factory(
        cls, mjx_model: mjx.Model, control_timestep: float = 0.01
    ) -> "SystemDynamicsMuJoCo":
        """
        Factory method to create a SystemDynamicsMuJoCo instance.
        Args:
            mjx_model (mjx.Model): The MuJoCo model to use for dynamics.
            control_timestep (float): The sample-hold time of actions.
                This must be a multiple of the simulation timestep.
        """
        state_dim = mjx_model.nq + mjx_model.nv
        action_dim = mjx_model.nu

        sim_per_control_step = int(
            jnp.round(control_timestep / mjx_model.opt.timestep)
        )

        assert jnp.allclose(
            control_timestep, sim_per_control_step * mjx_model.opt.timestep
        ), "control_timestep must be a multiple of the simulation timestep."

        return SystemDynamicsMuJoCo(
            state_dim=state_dim,
            action_dim=action_dim,
            mjx_model=mjx_model,
            sim_per_control_step=sim_per_control_step,
        )

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """Computes the next state given the current state and action.

        Uses vmap for efficient batch processing.

        Args:
            state (jax.Array): Current state, shape (..., state_dim).
            action (jax.Array): Current action, shape (..., action_dim).

        Returns:
            jax.Array: Next state, shape (..., state_dim).
        """
        # Store original shape to restore it later
        original_shape = state.shape
        batch_shape = original_shape[:-1]
        state_dim = state.shape[-1]
        action_dim = action.shape[-1]

        # Handle single state-action pair
        if not batch_shape:
            return self._compute_single_step(state, action)

        # Reshape to 2D for batch processing
        state_flat = state.reshape(-1, state_dim)
        action_flat = action.reshape(-1, action_dim)

        # Use vmap for batch processing
        next_state_flat = jax.vmap(self._compute_single_step)(
            state_flat, action_flat
        )

        # Reshape back to original batch shape
        return next_state_flat.reshape(*batch_shape, state_dim)

    def _normalize_quaternions(self, qpos: jax.Array) -> jax.Array:
        """Normalize quaternions in qpos array for FREE and BALL joints.

        Args:
            qpos (jax.Array): Position coordinates array.

        Returns:
            jax.Array: qpos with normalized quaternions.
        """
        # Work with a copy to avoid modifying the input
        normalized_qpos = jnp.array(qpos)

        # Small tolerance to avoid division by zero
        eps = 1e-8
        # Default identity quaternion [w, x, y, z]
        identity_quat = jnp.array([1.0, 0.0, 0.0, 0.0])

        # Iterate through all joints to find quaternion positions
        for i in range(self.mjx_model.njnt):
            joint_type = self.mjx_model.jnt_type[i]

            # FREE joint (type 0) has quaternion at positions [3:7]
            # relative to joint start
            # BALL joint (type 1) has quaternion at positions [0:4]
            # relative to joint start
            if joint_type == 0:  # FREE joint
                # Skip 3 translation coords
                quat_start = self.mjx_model.jnt_qposadr[i] + 3
                quat_end = quat_start + 4
                if quat_end <= len(qpos):
                    quat = qpos[quat_start:quat_end]
                    quat_norm = jnp.linalg.norm(quat)
                    # Avoid division by zero and normalize
                    quat_normalized = jnp.where(
                        quat_norm > eps, quat / quat_norm, identity_quat
                    )
                    normalized_qpos = normalized_qpos.at[
                        quat_start:quat_end
                    ].set(quat_normalized)

            elif joint_type == 1:  # BALL joint
                quat_start = self.mjx_model.jnt_qposadr[i]
                quat_end = quat_start + 4
                if quat_end <= len(qpos):
                    quat = qpos[quat_start:quat_end]
                    quat_norm = jnp.linalg.norm(quat)
                    # Avoid division by zero and normalize
                    quat_normalized = jnp.where(
                        quat_norm > eps, quat / quat_norm, identity_quat
                    )
                    normalized_qpos = normalized_qpos.at[
                        quat_start:quat_end
                    ].set(quat_normalized)

        return normalized_qpos

    def _compute_single_step(
        self, state: jax.Array, action: jax.Array
    ) -> jax.Array:
        """Simulates one state-action pair for sim_per_control_step steps.

        Args:
            state (jax.Array): Single state, shape (state_dim,).
            action (jax.Array): Single action, shape (action_dim,).

        Returns:
            jax.Array: Next state, shape (state_dim,).
        """

        data = mjx.make_data(self.mjx_model)

        # Normalize quaternions in incoming state for numerical stability
        qpos_input = state[: self.mjx_model.nq]
        qpos_normalized = self._normalize_quaternions(qpos_input)

        data = data.replace(
            qpos=qpos_normalized,
            qvel=state[self.mjx_model.nq :],
            ctrl=action,
        )

        data, _ = jax.lax.scan(
            lambda data, _: (mjx.step(self.mjx_model, data), None),
            init=data,
            xs=None,
            length=self.sim_per_control_step,
        )

        return jnp.concatenate([data.qpos, data.qvel])
