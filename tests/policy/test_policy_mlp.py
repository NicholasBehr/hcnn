import jax
import pytest
from flax import nnx

from hcnn.policy.policy_mlp import PolicyMLP
from hcnn.training.dropout_mlp import DropoutMLP
from tests.policy.test_policy_base import TestPolicyBase


class TestPolicyMLP(TestPolicyBase):
    """Test class for PolicyMLP."""

    @pytest.fixture
    def policy(self, state_dim, action_dim) -> PolicyMLP:
        key = jax.random.key(0)
        return PolicyMLP.factory(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layer_sizes=[5, 5],
            dropout_rate=0.2,
            rngs=nnx.Rngs(key),
        )

    @pytest.fixture(autouse=True)
    def setup(
        self,
        policy: PolicyMLP,
        state_dim: int,
        action_dim: int,
    ):
        self.policy = policy
        self.state_dim = state_dim
        self.action_dim = action_dim

    def test_instance_creation(self):
        assert isinstance(self.policy, PolicyMLP)
        assert self.policy.state_dim == self.state_dim
        assert self.policy.action_dim == self.action_dim
        assert isinstance(self.policy.dropout_mlp, DropoutMLP)
