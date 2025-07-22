import pytest

from hcnn.projection import ProjectionNoSubsNoTerm
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
