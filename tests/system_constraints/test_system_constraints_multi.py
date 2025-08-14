import jax.numpy as jnp
import pytest

from hcnn.system_constraints import (
    SystemConstraintsBall,
    SystemConstraintsBox,
    SystemConstraintsMulti,
)

from .test_system_constraints_base import TestSystemConstraintsBase


class TestSystemConstraintsMulti(TestSystemConstraintsBase):
    """Test class for SystemConstraintsMulti."""

    state_dim: int = 3
    action_dim: int = 2

    @pytest.fixture
    def sys_constraints(self) -> SystemConstraintsMulti:
        # Box constraint
        box = SystemConstraintsBox.factory(
            state_l=jnp.array([-1.0, -jnp.inf, -2.0]),
            state_u=jnp.array([+1.0, +jnp.inf, +2.0]),
            action_l=jnp.array([-0.5, -jnp.inf]),
            action_u=jnp.array([+0.5, +jnp.inf]),
        )

        # Ball exclusion in state space (action_dim must match for composition)
        ball = SystemConstraintsBall.factory(
            center=jnp.zeros((self.state_dim,)),
            radius=0.5,
            action_dim=self.action_dim,
        )

        return SystemConstraintsMulti.factory([box, ball])

    def test_instance_creation_multi(
        self, sys_constraints: SystemConstraintsMulti
    ):
        assert isinstance(sys_constraints, SystemConstraintsMulti)
        assert sys_constraints.state_dim == self.state_dim
        assert sys_constraints.action_dim == self.action_dim

    # Box contributes 3 active constraints; Ball contributes 1; total = 4
    assert sys_constraints.constraint_dim == 4

    def test_eval_shapes(self, sys_constraints: SystemConstraintsMulti):
        state = jnp.zeros((self.state_dim,))
        action = jnp.zeros((self.action_dim,))
        out = sys_constraints(state, action)
        assert out.shape == (sys_constraints.constraint_dim,)

        bs, t = 7, 5
        states = jnp.zeros((bs, t, self.state_dim))
        actions = jnp.zeros((bs, t, self.action_dim))
        out_batched = sys_constraints(states, actions)
        assert out_batched.shape == (bs, t, sys_constraints.constraint_dim)
