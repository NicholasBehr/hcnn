import jax
import jax.numpy as jnp
import pytest
from flax import nnx, struct

from hcnn.policy import PolicyBase
from hcnn.system_dynamics import SystemDynamicsBase
from tests.policy.policy_base import *


@struct.dataclass
class PolicyMock(PolicyBase):
    params: nnx.Param

    @nnx.jit
    def __call__(
        self,
        state: jax.Array,
        *,
        jax_key: jax.Array | None = None,
    ) -> jax.Array:
        return state @ self.params


@pytest.fixture
def policy() -> PolicyMock:
    return PolicyMock(
        state_dim=3,
        action_dim=2,
        params=nnx.Param(
            jnp.ones((3, 2)),
        ),
    )


@struct.dataclass
class SystemDynamicsMock(SystemDynamicsBase):
    params: nnx.Param

    @jax.jit
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        return state


@pytest.fixture
def instance() -> PolicyMock:
    return PolicyMock(
        state_dim=3,
        action_dim=2,
        params=nnx.Param(
            jnp.ones((3, 2)),
        ),
    )
