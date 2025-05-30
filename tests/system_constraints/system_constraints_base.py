import jax
import jax.numpy as jnp

from hcnn.system_constraints import SystemConstraintsBase


def test_call_dim(instance: SystemConstraintsBase):
    key = jax.random.key(0)
    if instance.state_dim is None:
        state = jnp.ones((1,))
    else:
        state = jax.random.normal(key, (instance.state_dim,))

    if instance.action_dim is None:
        action = jnp.ones((1,))
    else:
        action = jax.random.normal(key, (instance.action_dim,))

    constraint_eval = instance(state, action)

    assert constraint_eval.shape == (instance.constraint_dim,)


def test_lower_bound(instance: SystemConstraintsBase):
    assert jnp.ndim(instance.lower_bound) == 1
    assert jnp.issubdtype(instance.lower_bound.dtype, jnp.floating)
    assert instance.lower_bound.shape == (instance.constraint_dim,)


def test_upper_bound(instance: SystemConstraintsBase):
    assert jnp.ndim(instance.upper_bound) == 1
    assert jnp.issubdtype(instance.upper_bound.dtype, jnp.floating)
    assert instance.upper_bound.shape == (instance.constraint_dim,)
