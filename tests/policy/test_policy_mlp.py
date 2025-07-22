import jax
import pytest
from flax import nnx

from hcnn.policy import PolicyMLP
from hcnn.training.dropout_mlp import DropoutMLP

from .test_policy_base import TestPolicyBase


class TestPolicyMLP(TestPolicyBase):
    """Test class for PolicyMLP."""

    @pytest.fixture
    def policy(self) -> PolicyMLP:
        key = jax.random.key(0)
        return PolicyMLP.factory(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_features=[5, 5],
            dropout_rate=0.2,
            rngs=nnx.Rngs(key),
        )

    def test_instance_creation(self, policy: PolicyMLP):
        assert isinstance(policy, PolicyMLP)
        assert policy.state_dim == self.state_dim
        assert policy.action_dim == self.action_dim
        assert isinstance(policy.dropout_mlp, DropoutMLP)
