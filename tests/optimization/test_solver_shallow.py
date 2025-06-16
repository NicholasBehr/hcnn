import pytest

from hcnn.optimization import OptimizationSolverShallow
from tests.optimization.optimization_base import *


@pytest.fixture
def solver():
    return OptimizationSolverShallow(tol=1e-5, max_iter=1000)
