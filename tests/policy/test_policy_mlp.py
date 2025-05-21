import jax
import pytest
from flax import nnx

from hcnn.policy import PolicyMLP
from tests.policy.policy_base import *


@pytest.fixture
def instance() -> PolicyMLP:
    key = jax.random.key(0)
    return PolicyMLP.factory(
        state_dim=4,
        action_dim=2,
        hidden_layer_sizes=[5, 5],
        dropout_rate=0.2,
        rngs=nnx.Rngs(key),
    )
