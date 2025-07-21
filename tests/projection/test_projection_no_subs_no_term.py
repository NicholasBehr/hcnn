import jax
import jax.numpy as jnp
import mujoco
import pytest
from flax import nnx

from hcnn.projection import ProjectionNoSubsNoTerm, ProjectionPhiND
from hcnn.system_dynamics import SystemDynamicsMuJoCo
from tests.projection.test_projection_base import TestProjectionBase


class TestProjectionNoSubstitutionNoTerminal(TestProjectionBase):
    """Test class for ProjectionNoSubstitutionNoTerminal."""

    @pytest.fixture
    def projection(
        self,
        policy,
        relax_to_go,
        sys_dynamics,
        sys_constraints,
    ) -> ProjectionNoSubsNoTerm:
        return ProjectionNoSubsNoTerm.factory(
            policy=policy,
            relax_to_go=relax_to_go,
            sys_dynamics=sys_dynamics,
            sys_constraints=sys_constraints,
            horizon=self.horizon,
            relaxation_penalty=1.0,
        )


class TestProjectionNoSubstitutionNoTerminalMuJoCo(TestProjectionBase):
    """Test class for ProjectionNoSubstitutionNoTerminal."""

    control_timestep: float = 0.1
    state_dim: int = 6
    action_dim: int = 1
    constraint_dim: int = 7

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

    @pytest.fixture
    def projection(
        self,
        policy,
        relax_to_go,
        sys_dynamics,
        sys_constraints,
    ) -> ProjectionNoSubsNoTerm:
        return ProjectionNoSubsNoTerm.factory(
            policy=policy,
            relax_to_go=relax_to_go,
            sys_dynamics=sys_dynamics,
            sys_constraints=sys_constraints,
            horizon=self.horizon,
            relaxation_penalty=1.0,
        )

    @pytest.fixture
    def phi_nd(self, policy, relax_to_go) -> ProjectionPhiND:
        return ProjectionPhiND(
            policy_non_params=nnx.state(policy, nnx.Not(nnx.Param)),
            relax_to_go_non_params=nnx.state(relax_to_go, nnx.Not(nnx.Param)),
            initial_state=jnp.array(
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            jax_key=jax.random.PRNGKey(0),
        )
