import jax
import jax.numpy as jnp
from flax import struct
from mujoco import mjx

from hcnn.system_dynamics.system_dynamics_base import SystemDynamicsBase


@struct.dataclass
class SystemDynamicsMuJoCo(SystemDynamicsBase):
    mjx_model: mjx.Model
    sim_per_control_step: int

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
        def multi_step_fn(state: jax.Array, action: jax.Array) -> jax.Array:
            """Simulates one state-action pair for sim_per_control_step."""

            def step_fn(data: mjx.Data, _) -> tuple[mjx.Data, None]:
                """Helper function to step the simulation for one mjx.Data."""
                return mjx.step(self.mjx_model, data), None

            data = mjx.make_data(self.mjx_model)
            data = data.replace(
                qpos=state[: self.mjx_model.nq],
                qvel=state[self.mjx_model.nq :],
                ctrl=action,
            )

            data, _ = jax.lax.scan(
                step_fn,
                init=data,
                xs=None,
                length=self.sim_per_control_step,
            )
            return jnp.concatenate([data.qpos, data.qvel])

        # Store original shape to restore it later.
        original_shape = state.shape
        batch_shape = original_shape[:-1]

        # If there are no batch dimensions, just call the function.
        if not batch_shape:
            return multi_step_fn(state, action)

        # Reshape state and action to be 2D (batch_size, dim).
        state_reshaped = state.reshape(-1, self.state_dim)
        action_reshaped = action.reshape(-1, self.action_dim)

        # Vmap over the batch dimension.
        next_state_reshaped = jax.vmap(multi_step_fn)(
            state_reshaped, action_reshaped
        )

        # Reshape the output to the original batch shape.
        return next_state_reshaped.reshape(*batch_shape, self.state_dim)
