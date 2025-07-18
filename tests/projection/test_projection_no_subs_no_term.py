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

    control_timestep: float = 0.01
    state_dim: int = 13
    action_dim: int = 3
    constraint_dim: int = 16

    model_xml = r"""
        <mujoco>
            <option timestep="0.01"/>
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
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            jax_key=jax.random.PRNGKey(0),
        )
