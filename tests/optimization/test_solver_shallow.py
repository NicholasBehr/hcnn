import pytest

from hcnn.optimization.solver_shallow import OptimizationSolverShallow
from tests.optimization.optimization_base import *  # noqa: F403


@pytest.fixture
def solver():
    return OptimizationSolverShallow(tol=1e-5, max_iter=5_000)
