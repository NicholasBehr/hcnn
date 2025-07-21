import jax
import jax.numpy as jnp
from flax import struct
from mujoco import mjx

from hcnn.system_costs.system_costs_base import SystemCostsBase


@struct.dataclass
class SystemCostsMuJoCo(SystemCostsBase):
    """System costs for a MuJoCo model.
    This class computes costs based on the position and orientation of bodies
    in a MuJoCo model, using per-body cost weights and reference states.
    Note that positions and rotations are in world coordinates!!
    """

    model: mjx.Model
    cost_position: jax.Array
    cost_orientation: jax.Array
    position_reference: jax.Array
    quaternion_reference: jax.Array

    @classmethod
    def factory(
        cls,
        model: mjx.Model,
        cost_position: jax.Array,
        cost_orientation: jax.Array,
        position_reference: jax.Array,
        quaternion_reference: jax.Array,
    ):
        """Initializes the system costs.
        Cost and reference arrays must match the number of bodies in the model,
        including the world body.
        Note that all positions and rotations are in world coordinates!!

        Args:
            model (mjx.Model): The MuJoCo model.
            cost_position (jax.Array): Cost weight for body positions,
                shape (nbody,).
            cost_orientation (jax.Array): Cost weight for body orientation,
                shape (nbody,).
            position_reference (jax.Array): Reference positions,
                shape (nbody, 3).
            quaternion_reference (jax.Array): Reference unit quaternions,
                shape (nbody, 4).
        """
        state_dim = model.nq + model.nv
        action_dim = model.nu

        assert cost_position.shape == (model.nbody,)
        assert cost_orientation.shape == (model.nbody,)
        assert position_reference.shape == (model.nbody, 3)
        assert quaternion_reference.shape == (model.nbody, 4)

        # Verify that the reference quaternions are unit quaternions.
        assert jnp.all(
            jnp.isclose(jnp.linalg.norm(quaternion_reference, axis=-1), 1.0)
        ), "Reference quaternions must be unit quaternions."

        return SystemCostsMuJoCo(
            state_dim=state_dim,
            action_dim=action_dim,
            model=model,
            cost_position=cost_position,
            cost_orientation=cost_orientation,
            position_reference=position_reference,
            quaternion_reference=quaternion_reference,
        )

    @jax.jit
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """Computes the cost given the current state and action.

        The cost is a combination of position and orientation costs for
        each body. Uses vmap for efficient batch processing.

        Args:
            state (jax.Array): Current state, shape (..., state_dim).
            action (jax.Array): Current action, shape (..., action_dim).

        Returns:
            jax.Array: Stage cost, shape (...,).
        """
        # Store original shape to restore it later
        original_shape = state.shape
        batch_shape = original_shape[:-1]
        state_dim = state.shape[-1]
        action_dim = action.shape[-1]

        # Handle single state-action pair
        if not batch_shape:
            return self._compute_single_cost(state, action)

        # Reshape to 2D for batch processing
        state_flat = state.reshape(-1, state_dim)
        action_flat = action.reshape(-1, action_dim)

        # Use vmap for batch processing
        cost_flat = jax.vmap(self._compute_single_cost)(state_flat, action_flat)

        # Reshape back to original batch shape
        return cost_flat.reshape(*batch_shape)

    def _compute_single_cost(
        self, state: jax.Array, action: jax.Array
    ) -> jax.Array:
        """Computes cost for a single state-action pair.

        Args:
            state (jax.Array): Single state, shape (state_dim,).
            action (jax.Array): Single action, shape (action_dim,).

        Returns:
            jax.Array: Single cost value, shape ().
        """
        # Create MJX data for single state-action pair
        data = mjx.make_data(self.model)

        # Extract qpos and qvel from state
        qpos = jnp.asarray(state[: self.model.nq])
        qvel = jnp.asarray(state[self.model.nq :])

        # Set data - ensure arrays are JAX arrays
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.asarray(action))

        # Forward kinematics
        data = mjx.forward(self.model, data)

        # Position cost computation (include all bodies)
        # data.xpos shape: (nbody, 3)
        # self.cost_position shape: (nbody,)
        # self.position_reference shape: (nbody, 3)
        position_error = data.xpos - self.position_reference
        # Compute squared error per body, then weight by per-body costs
        position_error_squared = jnp.sum(position_error**2, axis=1)  # (nbody,)
        position_cost = jnp.sum(self.cost_position * position_error_squared)

        # Orientation cost computation (include all bodies)
        # data.xquat shape: (nbody, 4)
        # self.quaternion_reference shape: (nbody, 4)
        q_ref = self.quaternion_reference
        q_cur = data.xquat

        # Ensure shortest path rotation (q and -q represent same orientation)
        dot_product = jnp.sum(q_ref * q_cur, axis=-1, keepdims=True)
        q_cur = jnp.where(dot_product < 0, -q_cur, q_cur)

        # The error quaternion is q_cur * q_ref_conjugate
        # For q = [w, x, y, z], conjugate is [w, -x, -y, -z]
        # MJX uses [w, x, y, z] quaternion format
        q_ref_conj = q_ref * jnp.array([1.0, -1.0, -1.0, -1.0])

        # Hamilton product for q_error = q_cur * q_ref_conj
        w_cur, v_cur = q_cur[..., 0], q_cur[..., 1:]
        w_ref, v_ref = q_ref_conj[..., 0], q_ref_conj[..., 1:]
        w_err = w_cur * w_ref - jnp.sum(v_cur * v_ref, axis=-1)

        # Calculate the angle of rotation from the error quaternion
        # Clamp w_err to avoid domain errors with acos
        w_err_clamped = jnp.clip(w_err, -1.0, 1.0)
        angle = 2 * jnp.arccos(jnp.abs(w_err_clamped))

        # Sum over all bodies using per-body cost weights
        # angle shape: (nbody,)
        # self.cost_orientation shape: (nbody,)
        orientation_cost = jnp.sum(self.cost_orientation * angle**2)

        return position_cost + orientation_cost
