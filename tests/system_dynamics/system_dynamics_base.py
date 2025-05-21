import jax

from hcnn.system_dynamics import SystemDynamicsBase


def test_next_state_dimensions(instance: SystemDynamicsBase):
    key = jax.random.key(0)
    state = jax.random.normal(key, (instance.state_dim,))
    action = jax.random.normal(key, (instance.action_dim,))

    next_state = instance.next_state(state, action)
    assert next_state.shape == (instance.state_dim,)
