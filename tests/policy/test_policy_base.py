import jax
import pytest
from flax import nnx, struct

from hcnn.policy import PolicyBase


@struct.dataclass
class PolicyMock(PolicyBase):
    gain: nnx.Param

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
            gain=nnx.Param(jax.random.uniform(key, (state_dim, action_dim))),
        )

    def __call__(
        self,
        state: jax.Array,
        *,
        jax_key: jax.Array | None = None,
    ) -> jax.Array:
        return state @ self.gain


class TestPolicyBase:
    """Test class for PolicyBase."""

    state_dim: int = 10
    action_dim: int = 5

    @pytest.fixture
    def policy(self) -> PolicyBase:
        return PolicyMock.factory(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )

    def test_instance_creation_base(self, policy: PolicyBase):
        assert isinstance(policy, PolicyBase)
        assert isinstance(policy.state_dim, int)
        assert isinstance(policy.action_dim, int)

    def test_call_single_w_key(self, policy: PolicyBase):
        key = jax.random.key(0)
        state = jax.random.normal(key, (self.state_dim,))

        policy_action = policy(state, jax_key=key)
        assert policy_action.shape == (self.action_dim,)

    def test_call_batch_w_key(self, policy: PolicyBase):
        batch_size = 200
        horizon = 20
        key = jax.random.key(0)
        state = jax.random.normal(key, (batch_size, horizon, self.state_dim))

        policy_action = policy(state, jax_key=key)
        assert policy_action.shape == (batch_size, horizon, self.action_dim)

    def test_call_single_wo_key(self, policy: PolicyBase):
        key = jax.random.key(0)
        state = jax.random.normal(key, (self.state_dim,))

        policy_action = policy(state)
        assert policy_action.shape == (self.action_dim,)

    def test_call_batch_wo_key(self, policy: PolicyBase):
        batch_size = 200
        horizon = 20
        key = jax.random.key(0)
        state = jax.random.normal(key, (batch_size, horizon, self.state_dim))

        policy_action = policy(state)
        assert policy_action.shape == (batch_size, horizon, self.action_dim)

    def test_call_jit(self, policy: PolicyBase):
        key = jax.random.key(0)
        state = jax.random.normal(key, (self.state_dim,))

        jitted_call = nnx.jit(lambda p, s: p(s))
        action = jitted_call(policy, state)
        assert action.shape == (self.action_dim,)
