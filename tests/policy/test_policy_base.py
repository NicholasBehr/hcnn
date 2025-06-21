import jax
import pytest
from flax import nnx, struct

from hcnn.policy.policy_base import PolicyBase


@struct.dataclass
class PolicyMock(PolicyBase):
    params: nnx.Param

    @classmethod
    def factory(
        cls,
        state_dim: int,
        action_dim: int,
    ) -> "PolicyMock":
        key = jax.random.key(0)
        return cls(
            state_dim=state_dim,
            action_dim=action_dim,
            params=jax.random.normal(key, (state_dim, action_dim)),
        )

    def __call__(
        self,
        state: jax.Array,
        *,
        jax_key: jax.Array | None = None,
    ) -> jax.Array:
        return state @ self.params


class TestPolicyBase:
    """Test class for PolicyBase."""

    @pytest.fixture
    def policy(self, state_dim, action_dim) -> PolicyBase:
        return PolicyMock.factory(
            state_dim=state_dim,
            action_dim=action_dim,
        )

    @pytest.fixture(autouse=True)
    def setup(self, policy: PolicyBase):
        self.policy = policy
        self.state_dim = policy.state_dim
        self.action_dim = policy.action_dim

    def test_instance_creation(self):
        assert isinstance(self.policy, PolicyBase)
        assert isinstance(self.policy.state_dim, int)
        assert isinstance(self.policy.action_dim, int)

    def test_call_single_w_key(self):
        key = jax.random.key(0)
        state = jax.random.normal(key, (self.state_dim,))

        policy_action = self.policy(state, jax_key=key)
        assert policy_action.shape == (self.action_dim,)

    def test_call_batch_w_key(self):
        batch_size = 200
        horizon = 20
        key = jax.random.key(0)
        state = jax.random.normal(key, (batch_size, horizon, self.state_dim))

        policy_action = self.policy(state, jax_key=key)
        assert policy_action.shape == (batch_size, horizon, self.action_dim)

    def test_call_single_wo_key(self):
        key = jax.random.key(0)
        state = jax.random.normal(key, (self.state_dim,))

        policy_action = self.policy(state)
        assert policy_action.shape == (self.action_dim,)

    def test_call_batch_wo_key(self):
        batch_size = 200
        horizon = 20
        key = jax.random.key(0)
        state = jax.random.normal(key, (batch_size, horizon, self.state_dim))

        policy_action = self.policy(state)
        assert policy_action.shape == (batch_size, horizon, self.action_dim)
