import jax
import jax.numpy as jnp
import pytest
from flax import struct

from hcnn.system_costs import SystemCostsMulti
from hcnn.system_costs.system_costs_base import SystemCostsBase

from .test_system_costs_base import TestSystemCostBase


@struct.dataclass
class SystemCostMock(SystemCostsBase):
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        return jnp.sum(state, axis=-1) + jnp.sum(action, axis=-1)


class TestSystemCostsMulti(TestSystemCostBase):
    state_dim: int = 4
    action_dim: int = 2

    @pytest.fixture
    def multi_cost(self, sys_cost: SystemCostsBase):
        return SystemCostsMulti.factory(sys_cost, sys_cost)

    def test_instance_creation_multi(self, multi_cost: SystemCostsMulti):
        assert isinstance(multi_cost, SystemCostsMulti)
        assert len(multi_cost.costs) == 2

    def test_factory_dimension_mismatch(self):
        cost1 = SystemCostMock(state_dim=4, action_dim=2)
        cost2 = SystemCostMock(state_dim=5, action_dim=2)
        with pytest.raises(ValueError):
            SystemCostsMulti.factory(cost1, cost2)

    def test_factory_empty(self):
        with pytest.raises(ValueError):
            SystemCostsMulti.factory()

    def test_call_example(
        self, multi_cost: SystemCostsMulti, sys_cost: SystemCostsBase
    ):
        state = jax.random.normal(jax.random.PRNGKey(0), (10, self.state_dim))
        action = jax.random.normal(jax.random.PRNGKey(1), (10, self.action_dim))
        result = multi_cost(state, action)
        assert result.shape == (10,)
        assert jnp.issubdtype(result.dtype, jnp.floating)

        expected_result = sys_cost(state, action) + sys_cost(state, action)
        assert jnp.allclose(result, expected_result)
