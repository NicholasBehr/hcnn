import jax
import jax.numpy as jnp
import pytest
from flax import nnx, struct
from jax import flatten_util

from hcnn.policy import PolicyBase
from hcnn.projection import ProjectionBase, ProjectionPhi, Rollout
from hcnn.system_constraints import SystemConstraintsBase
from hcnn.system_dynamics import SystemDynamicsBase

state_dim = 3
action_dim = 2


### POLICY MOCK ###
@struct.dataclass
class PolicyMock(PolicyBase):
    params: nnx.Param

    def __call__(
        self,
        state: jax.Array,
        *,
        jax_key: jax.Array | None = None,
    ) -> jax.Array:
        return state @ self.params


@pytest.fixture
def policy() -> PolicyBase:
    return PolicyMock(
        state_dim=state_dim,
        action_dim=action_dim,
        params=nnx.Param(
            jnp.ones((state_dim, action_dim)),
        ),
    )


### SYSTEM DYNAMICS MOCK ###
@struct.dataclass
class SystemDynamicsMock(SystemDynamicsBase):
    @jax.jit
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        return state


@pytest.fixture
def sys_dynamics() -> SystemDynamicsBase:
    return SystemDynamicsMock(
        state_dim=state_dim,
        action_dim=action_dim,
    )


### SYSTEM CONSTRAINTS MOCK ###
@struct.dataclass
class SystemConstraintsMock(SystemConstraintsBase):
    @jax.jit
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        return jnp.concatenate([state, action], axis=-1)


@pytest.fixture
def sys_constraints() -> SystemConstraintsBase:
    return SystemConstraintsMock(
        state_dim=state_dim,
        action_dim=action_dim,
        constraint_dim=state_dim + action_dim,
        lower_bound=jnp.zeros(state_dim + action_dim),
        upper_bound=jnp.ones(state_dim + action_dim),
    )


### POJECTION PHI ###
@pytest.fixture
def phi(policy) -> ProjectionPhi:
    return ProjectionPhi(
        initial_state=jnp.zeros((policy.state_dim,)),
        jax_key=jax.random.key(0),
        policy_params=nnx.state(policy, nnx.Param),
    )


def test_instance_creation(instance):
    assert isinstance(instance, ProjectionBase)
    assert instance.state_dim == state_dim
    assert instance.action_dim == action_dim
    assert isinstance(instance.horizon, int)

    # Check the policy graphdef and non_params
    assert isinstance(instance.policy_graphdef, nnx.GraphDef)
    assert isinstance(instance.policy_non_params, nnx.State)

    # Check the split dynamics and constraints
    assert isinstance(instance.split_dynamics[0], nnx.GraphDef)
    assert isinstance(instance.split_dynamics[1], nnx.State)
    assert isinstance(instance.split_constraints[0], nnx.GraphDef)
    assert isinstance(instance.split_constraints[1], nnx.State)


def test_x_to_rollout(instance):
    key = jax.random.key(0)
    # Create a dummy input array
    x = jax.random.normal(key, instance.cost_quad_diag.shape)

    # Call the x_to_rollout method
    rollout = instance.x_to_rollout(x)

    # Check the structure of the returned Rollout
    assert isinstance(rollout, Rollout)
    assert rollout.states.shape == (instance.horizon, instance.state_dim)
    assert rollout.actions.shape == (instance.horizon, instance.action_dim)
    assert rollout.interventions.shape == (
        instance.horizon,
        instance.action_dim,
    )
    assert rollout.relaxations_plus.shape == (instance.horizon,)
    assert rollout.relaxations_minus.shape == (instance.horizon,)

    # Check that this alligs with ravel pytree
    assert jnp.array_equal(flatten_util.ravel_pytree(rollout)[0], x)
