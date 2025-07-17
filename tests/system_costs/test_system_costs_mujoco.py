import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx
import pytest

from hcnn.system_costs import SystemCostsMuJoCo

from .test_system_costs_base import TestSystemCostBase


class TestSystemCostsMuJoCo(TestSystemCostBase):
    # For freejoint: nq=7, nv=6
    state_dim: int = 13
    action_dim: int = 3

    model_xml = r"""
        <mujoco>
            <option timestep="0.02"/>
            <worldbody>
                <body>
                <freejoint name="freejoint"/>
                <geom size=".15" mass="1" type="sphere"/>
                </body>
            </worldbody>
            <actuator>
                <motor joint="freejoint" gear="0 0 0 1 0 0"/>
                <motor joint="freejoint" gear="0 0 0 0 1 0"/>
                <motor joint="freejoint" gear="0 0 0 0 0 1"/>
            </actuator>
        </mujoco>
        """

    @pytest.fixture
    def model(self):
        return mujoco.MjModel.from_xml_string(self.model_xml)

    @pytest.fixture
    def mjx_model(self, model):
        return mujoco.mjx.put_model(model)

    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def cost_position(self, model, key):
        # Cost per body (include world body)
        key_pos, _ = jax.random.split(key)
        return jax.random.uniform(key_pos, (model.nbody,))

    @pytest.fixture
    def cost_orientation(self, model, key):
        # Cost per body (include world body)
        _, key_ang = jax.random.split(key)
        return jax.random.uniform(key_ang, (model.nbody,))

    @pytest.fixture
    def position_reference(self, model):
        # Zero position reference for all bodies (include world body)
        return jnp.zeros((model.nbody, 3))

    @pytest.fixture
    def quaternion_reference(self, model):
        # Use identity quaternion for reference, include world body
        q_ref = jnp.array([1.0, 0.0, 0.0, 0.0])
        return jnp.tile(q_ref, (model.nbody, 1))

    @pytest.fixture
    def sys_cost(
        self,
        mjx_model,
        cost_position,
        cost_orientation,
        position_reference,
        quaternion_reference,
    ) -> SystemCostsMuJoCo:
        return SystemCostsMuJoCo.factory(
            model=mjx_model,
            cost_position=cost_position,
            cost_orientation=cost_orientation,
            position_reference=position_reference,
            quaternion_reference=quaternion_reference,
        )

    def test_instance_creation_mujoco(self, sys_cost: SystemCostsMuJoCo):
        assert isinstance(sys_cost, SystemCostsMuJoCo)
        assert sys_cost.state_dim == self.state_dim
        assert sys_cost.action_dim == self.action_dim

    def test_invalid_cost_shapes(
        self,
        mjx_model,
        cost_position,
        cost_orientation,
        position_reference,
        quaternion_reference,
    ):
        with pytest.raises(AssertionError):
            SystemCostsMuJoCo.factory(
                model=mjx_model,
                cost_position=cost_position[1:],  # Invalid shape
                cost_orientation=cost_orientation,
                position_reference=position_reference,
                quaternion_reference=quaternion_reference,
            )
        with pytest.raises(AssertionError):
            SystemCostsMuJoCo.factory(
                model=mjx_model,
                cost_position=cost_position,
                cost_orientation=cost_orientation[1:],  # Invalid shape
                position_reference=position_reference,
                quaternion_reference=quaternion_reference,
            )
        with pytest.raises(AssertionError):
            SystemCostsMuJoCo.factory(
                model=mjx_model,
                cost_position=cost_position,
                cost_orientation=cost_orientation,
                position_reference=position_reference[1:],  # Invalid shape
                quaternion_reference=quaternion_reference,
            )
        with pytest.raises(AssertionError):
            SystemCostsMuJoCo.factory(
                model=mjx_model,
                cost_position=cost_position,
                cost_orientation=cost_orientation,
                position_reference=position_reference,
                quaternion_reference=quaternion_reference[1:],  # Invalid shape
            )

    def test_position_cost(
        self, model, sys_cost: SystemCostsMuJoCo, cost_position
    ):
        """Test case for position cost with zero orientation cost."""
        data = mujoco.MjData(model)
        # Set qpos for a specific position and identity orientation
        data.qpos = jnp.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
        data.qvel = jnp.zeros(model.nv)
        state = jnp.concatenate([data.qpos, data.qvel])
        action = jnp.zeros(sys_cost.action_dim)

        mujoco.mj_forward(model, data)

        # Expected cost: position cost for all bodies
        # and orientation cost (identity quaternion = zero error)
        position_error = data.xpos - sys_cost.position_reference
        # Compute squared error per body, then weight by per-body costs
        position_error_squared = jnp.sum(position_error**2, axis=1)
        position_cost_ref = jnp.sum(cost_position * position_error_squared)
        # Orientation cost should be zero since we use identity quaternion
        cost_ref = position_cost_ref

        cost = sys_cost(state, action)
        assert jnp.allclose(cost, cost_ref)

    def _test_orientation_cost(
        self,
        model,
        sys_cost,
        cost_position,
        cost_orientation,
        q_rot,
        angle_error_vec,
    ):
        """Helper function to test orientation costs."""
        data = mujoco.MjData(model)
        # Set qpos for zero position and a given rotation
        data.qpos = jnp.concatenate([jnp.zeros(3), q_rot])
        data.qvel = jnp.zeros(model.nv)
        state = jnp.concatenate([data.qpos, data.qvel])
        action = jnp.zeros(sys_cost.action_dim)

        mujoco.mj_forward(model, data)

        # Expected cost: quadratic position and orientation costs
        # Position cost: include all bodies
        position_error = data.xpos - sys_cost.position_reference
        # Compute squared error per body, then weight by per-body costs
        position_error_squared = jnp.sum(position_error**2, axis=1)
        position_cost_ref = jnp.sum(cost_position * position_error_squared)
        # Orientation cost: per-body weights times squared angle per body
        angle_magnitude = jnp.linalg.norm(angle_error_vec)
        orientation_cost_ref = cost_orientation[1] * (angle_magnitude**2)
        cost_ref = position_cost_ref + orientation_cost_ref

        cost = sys_cost(state, action)
        assert jnp.allclose(cost, cost_ref, atol=1e-6)

    def test_orientation_cost_x_rotation(
        self,
        model,
        sys_cost: SystemCostsMuJoCo,
        cost_position,
        cost_orientation,
    ):
        """Test orientation cost for rotation around the x-axis."""
        angle_rad = jnp.pi / 2
        q_rot = jnp.array(
            [jnp.cos(angle_rad / 2), jnp.sin(angle_rad / 2), 0, 0]
        )
        angle_error_vec = jnp.array([angle_rad, 0.0, 0.0])
        self._test_orientation_cost(
            model,
            sys_cost,
            cost_position,
            cost_orientation,
            q_rot,
            angle_error_vec,
        )

    def test_orientation_cost_y_rotation(
        self,
        model,
        sys_cost: SystemCostsMuJoCo,
        cost_position,
        cost_orientation,
    ):
        """Test orientation cost for rotation around the y-axis."""
        angle_rad = jnp.pi / 2
        q_rot = jnp.array(
            [jnp.cos(angle_rad / 2), 0, jnp.sin(angle_rad / 2), 0]
        )
        angle_error_vec = jnp.array([0.0, angle_rad, 0.0])
        self._test_orientation_cost(
            model,
            sys_cost,
            cost_position,
            cost_orientation,
            q_rot,
            angle_error_vec,
        )

    def test_orientation_cost_z_rotation(
        self,
        model,
        sys_cost: SystemCostsMuJoCo,
        cost_position,
        cost_orientation,
    ):
        """Test orientation cost for rotation around the z-axis."""
        angle_rad = jnp.pi / 2
        q_rot = jnp.array(
            [jnp.cos(angle_rad / 2), 0, 0, jnp.sin(angle_rad / 2)]
        )
        angle_error_vec = jnp.array([0.0, 0.0, angle_rad])
        self._test_orientation_cost(
            model,
            sys_cost,
            cost_position,
            cost_orientation,
            q_rot,
            angle_error_vec,
        )

    def test_quaternion_format_validation(
        self, mjx_model, cost_position, cost_orientation, position_reference
    ):
        """Test that quaternion reference validation works correctly."""
        invalid_quat = jnp.array([[2.0, 0.0, 0.0, 0.0]])  # Not unit quaternion
        invalid_quat_ref = jnp.tile(invalid_quat, (mjx_model.nbody, 1))

        with pytest.raises(AssertionError, match="unit quaternions"):
            SystemCostsMuJoCo.factory(
                model=mjx_model,
                cost_position=cost_position,
                cost_orientation=cost_orientation,
                position_reference=position_reference,
                quaternion_reference=invalid_quat_ref,
            )
