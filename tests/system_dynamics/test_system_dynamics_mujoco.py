import jax.numpy as jnp
import mujoco
import pytest

from hcnn.system_dynamics import SystemDynamicsMuJoCo

from .test_system_dynamics_base import TestSystemDynamicsBase


class TestSystemDynamicsMuJoCo(TestSystemDynamicsBase):
    control_timestep: float = 0.1
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
    def sys_dynamics(self) -> SystemDynamicsMuJoCo:
        model = mujoco.MjModel.from_xml_string(self.model_xml)
        mjx_model = mujoco.mjx.put_model(model)

        return SystemDynamicsMuJoCo.factory(
            mjx_model=mjx_model,
            control_timestep=self.control_timestep,
        )

    def test_instance_creation_mujoco(self, sys_dynamics: SystemDynamicsMuJoCo):
        assert isinstance(sys_dynamics, SystemDynamicsMuJoCo)
        assert sys_dynamics.sim_per_control_step == 5

    def test_invalid_control_timestep(self):
        model = mujoco.MjModel.from_xml_string(self.model_xml)
        mjx_model = mujoco.mjx.put_model(model)

        with pytest.raises(AssertionError):
            SystemDynamicsMuJoCo.factory(
                mjx_model=mjx_model, control_timestep=0.03
            )

    def test_call_example_single(self, sys_dynamics: SystemDynamicsMuJoCo):
        model = mujoco.MjModel.from_xml_string(self.model_xml)
        data = mujoco.MjData(model)
        data.qpos = jnp.ones(model.nq)
        data.qvel = jnp.ones(model.nv)

        state = jnp.concatenate([data.qpos, data.qvel])
        action = jnp.array([0.1, 0.25, 0.3])
        data.ctrl = action

        for _ in range(sys_dynamics.sim_per_control_step):
            mujoco.mj_step(model, data)

        next_state_ref = jnp.concatenate([data.qpos, data.qvel])
        next_state = sys_dynamics(state, action)
        assert jnp.allclose(next_state, next_state_ref)
