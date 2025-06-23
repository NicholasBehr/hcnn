import pytest

from hcnn.optimization import OptimizationSolverShallow

from .optimization_base import *  # noqa: F403


@pytest.fixture
def solver():
    return OptimizationSolverShallow(tol=1e-5, max_iter=10_000)
