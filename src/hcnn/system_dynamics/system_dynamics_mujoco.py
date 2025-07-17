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
        data = data.replace(
            qpos=state[: self.mjx_model.nq],
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
