import jax

from hcnn.system_cost import SystemCostBase


def test_call_single(instance: SystemCostBase):
    key = jax.random.key(0)
    state = jax.random.normal(key, (instance.state_dim,))
    action = jax.random.normal(key, (instance.action_dim,))

    cost = instance(state, action)
    assert cost.shape == ()


def test_call_batch(instance: SystemCostBase):
    batch_size = 200
    horizon = 20
    key = jax.random.key(0)
    states = jax.random.normal(key, (batch_size, horizon, instance.state_dim))
    actions = jax.random.normal(key, (batch_size, horizon, instance.action_dim))

    costs = instance(states, actions)
    assert costs.shape == (
        batch_size,
        horizon,
    )
