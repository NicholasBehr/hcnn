import jax.numpy as jnp
import mujoco
import pytest

from hcnn.system_dynamics import SystemDynamicsMuJoCo

from .test_system_dynamics_base import TestSystemDynamicsBase


class TestSystemDynamicsMuJoCo(TestSystemDynamicsBase):
    control_timestep: float = 0.1
    state_dim: int = 6
    action_dim: int = 1

    model_xml = r"""
        <mujoco>
        <option
            timestep=".02"
            integrator="implicitfast" solver="CG" tolerance="1e-4">
            <flag contact="disable" constraint="disable" sensor="disable"/>
        </option>

        <default>
            <joint type="hinge" axis="0 -1 0" limited="false" range="-150 150"/>
            <geom type="capsule" size=".02"/>
        </default>

        <worldbody>
            <light pos="0 -.4 1"/>
            <camera name="fixed" pos="0 -1.5 0" xyaxes="1 0 0 0 0 1"/>

            <body name="red" pos="0 0 .2">
                <joint name="root" />
                <geom fromto="0 0 0 0 0 -.25" rgba="1 0 0 1"/>
                <body name="green" pos="0 0 -.25">
                    <joint/>
                    <geom fromto="0 0 0 0 0 -.25" rgba="0 1 0 1"/>
                    <body name="blue" pos="0 0 -.25">
                        <joint/>
                        <geom fromto="0 0 0 0 0 -.25" rgba="0 0 1 1"/>
                    </body>
                </body>
            </body>
        </worldbody>

        <actuator>
            <motor name="joint_motor" joint="root"/>
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

        state = jnp.concatenate([data.qpos, data.qvel])
        action = jnp.array([0.1])
        data.ctrl = action

        for _ in range(sys_dynamics.sim_per_control_step):
            mujoco.mj_step(model, data)

        next_state_ref = jnp.concatenate([data.qpos, data.qvel])
        next_state = sys_dynamics(state, action)

        assert jnp.allclose(next_state, next_state_ref)
