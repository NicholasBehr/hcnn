import jax

from hcnn.system_dynamics import SystemDynamicsBase


def test_call_single(instance: SystemDynamicsBase):
    key = jax.random.key(0)
    state = jax.random.normal(key, (instance.state_dim,))
    action = jax.random.normal(key, (instance.action_dim,))

    next_state = instance(state, action)
    assert next_state.shape == (instance.state_dim,)


def test_call_batch(instance: SystemDynamicsBase):
    batch_size = 200
    horizon = 20
    key = jax.random.key(0)
    states = jax.random.normal(key, (batch_size, horizon, instance.state_dim))
    actions = jax.random.normal(key, (batch_size, horizon, instance.action_dim))

    next_states = instance(states, actions)
    assert next_states.shape == (batch_size, horizon, instance.state_dim)
