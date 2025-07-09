import jax
import pytest


def pytest_configure(config):
    use_gpu = False  # Set to False to use CPU

    if use_gpu:
        # Set JAX to use GPU 3 by default
        jax.config.update("jax_cuda_visible_devices", "3")
        jax.config.update("jax_platform_name", "gpu")
    else:
        # Set JAX to use CPU by default
        jax.config.update("jax_platform_name", "cpu")


@pytest.fixture
def state_dim() -> int:
    return 7


@pytest.fixture
def action_dim() -> int:
    return 3


@pytest.fixture
def horizon() -> int:
    return 10
