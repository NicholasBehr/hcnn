import pytest

from hcnn.optimization import OptimizationSolverOSQP

from .optimization_base import *  # noqa: F403


@pytest.fixture
def solver():
    return OptimizationSolverOSQP(tol=1e-4, max_iter=200, gradient_clip=1e4)
