import pytest

from hcnn.optimization.solver_osqp import OptimizationSolverOSQP
from tests.optimization.optimization_base import *  # noqa: F403


@pytest.fixture
def solver():
    return OptimizationSolverOSQP(tol=1e-5, max_iter=1_000)
