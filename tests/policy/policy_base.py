import jax

from hcnn.policy import PolicyBase


def test_call_single_w_key(instance: PolicyBase):
    key = jax.random.key(0)
    state = jax.random.normal(key, (instance.state_dim,))

    policy_action = instance(state, jax_key=key)
    assert policy_action.shape == (instance.action_dim,)


def test_call_batch_w_key(instance: PolicyBase):
    batch_size = 200
    key = jax.random.key(0)
    state = jax.random.normal(key, (batch_size, instance.state_dim))

    policy_action = instance(state, jax_key=key)
    assert policy_action.shape == (batch_size, instance.action_dim)


def test_call_single_wo_key(instance: PolicyBase):
    key = jax.random.key(0)
    state = jax.random.normal(key, (instance.state_dim,))

    policy_action = instance(state)
    assert policy_action.shape == (instance.action_dim,)


def test_call_batch_wo_key(instance: PolicyBase):
    batch_size = 200
    key = jax.random.key(0)
    state = jax.random.normal(key, (batch_size, instance.state_dim))

    policy_action = instance(state)
    assert policy_action.shape == (batch_size, instance.action_dim)
