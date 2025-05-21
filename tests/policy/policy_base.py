import jax

from hcnn.policy import PolicyBase


def test_call_dim(instance: PolicyBase):
    key = jax.random.key(0)
    state = jax.random.normal(key, (instance.state_dim,))

    policy_action = instance(state, jax_key=key)

    assert policy_action.shape == (instance.action_dim,)
