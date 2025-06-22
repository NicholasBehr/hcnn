import jax
import pytest
from flax import nnx

from hcnn.optimization.solver_shallow import OptimizationSolverShallow
from hcnn.projection.projection_base import (
    ProjectionPhi,
)
from hcnn.training.dropout_mlp import DropoutMLP
from hcnn.training.loss_and_grad import LossAndGrad
from tests.policy.test_policy_base import PolicyBase, PolicyMock
from tests.projection.test_projection_base import ProjectionMock
from tests.system_constraints.test_system_constraints_base import (
    SystemConstraintsBase,
    SystemConstraintsMock,
)
from tests.system_cost.test_system_cost_base import (
    SystemCostBase,
    SystemCostMock,
)
from tests.system_dynamics.test_system_dynamics_base import (
    SystemDynamicsBase,
    SystemDynamicsMock,
)


@pytest.fixture
def policy(state_dim, action_dim) -> PolicyBase:
    return PolicyMock.factory(
        state_dim=state_dim,
        action_dim=action_dim,
    )


@pytest.fixture
def sys_dynamics(state_dim, action_dim) -> SystemDynamicsBase:
    return SystemDynamicsMock(
        state_dim=state_dim,
        action_dim=action_dim,
    )


@pytest.fixture
def sys_constraints(state_dim, action_dim) -> SystemConstraintsBase:
    return SystemConstraintsMock.factory(
        state_dim=state_dim,
        action_dim=action_dim,
    )


@pytest.fixture
def value_to_go(state_dim):
    return DropoutMLP(
        layer_sizes=(state_dim, 64, 1),
        dropout_rate=0.1,
        rngs=nnx.Rngs(42),
    )


@pytest.fixture
def relax_to_go(state_dim):
    return DropoutMLP(
        layer_sizes=(state_dim, 64, 1),
        dropout_rate=0.1,
        rngs=nnx.Rngs(42),
    )


@pytest.fixture
def projection(
    policy,
    relax_to_go,
    sys_dynamics,
    sys_constraints,
    horizon,
) -> ProjectionMock:
    return ProjectionMock.factory(
        policy=policy,
        relax_to_go=relax_to_go,
        sys_dynamics=sys_dynamics,
        sys_constraints=sys_constraints,
        horizon=horizon,
    )


@pytest.fixture
def sys_cost(state_dim, action_dim) -> SystemCostBase:
    return SystemCostMock(
        state_dim=state_dim,
        action_dim=action_dim,
    )


@pytest.fixture
def phi(
    policy,
    state_dim,
) -> ProjectionPhi:
    keys = jax.random.split(jax.random.key(0), 2)
    return ProjectionPhi.factory(
        initial_state=jax.random.normal(keys[0], (state_dim,)),
        policy=policy,
        jax_key=keys[1],
    )


@pytest.fixture
def solver():
    return OptimizationSolverShallow(tol=0.01)


@pytest.fixture
def loss_and_grad(
    projection,
    solver,
    sys_cost,
    value_to_go,
    relax_to_go,
) -> LossAndGrad:
    return LossAndGrad.factory(
        projection=projection,
        sys_cost=sys_cost,
        solver=solver,
        value_to_go=value_to_go,
        relax_to_go=relax_to_go,
    )


def test_policy_loss_single(loss_and_grad: LossAndGrad):
    horizon = 20
    key = jax.random.key(42)
    value_residuals = jax.random.normal(key, (horizon - 1,))
    rollout_interventions = jax.random.normal(key, (horizon,))

    loss = loss_and_grad.policy_loss(
        value_residuals=value_residuals,
        rollout_interventions=rollout_interventions,
        td_gamma=0.8,
        gae_lambda=0.99,
    )

    assert loss.shape == ()


def test_policy_loss_batch(loss_and_grad: LossAndGrad):
    batch_size = 200
    horizon = 20
    key = jax.random.key(42)
    value_residuals = jax.random.normal(
        key, (batch_size, batch_size, horizon - 1)
    )
    rollout_interventions = jax.random.normal(
        key, (batch_size, batch_size, horizon)
    )

    loss = loss_and_grad.policy_loss(
        value_residuals=value_residuals,
        rollout_interventions=rollout_interventions,
        td_gamma=0.8,
        gae_lambda=0.99,
    )

    assert loss.shape == ()


def test_value_to_go_loss_single(loss_and_grad: LossAndGrad):
    horizon = 20
    key = jax.random.key(42)
    td_residuals = jax.random.normal(key, (horizon - 1,))

    loss = loss_and_grad.value_to_go_loss(
        value_residuals=td_residuals,
        td_gamma=0.8,
        gae_lambda=0.99,
    )

    assert loss.shape == ()


def test_value_to_go_loss_batch(loss_and_grad: LossAndGrad):
    horizon = 20
    batch_size = 200
    key = jax.random.key(42)
    td_residuals = jax.random.normal(key, (batch_size, batch_size, horizon - 1))

    loss = loss_and_grad.value_to_go_loss(
        value_residuals=td_residuals,
        td_gamma=0.8,
        gae_lambda=0.99,
    )

    assert loss.shape == ()


def test_relax_to_go_loss_single(loss_and_grad: LossAndGrad):
    horizon = 20
    key = jax.random.key(42)
    td_residuals = jax.random.normal(key, (horizon - 1,))

    loss = loss_and_grad.relax_to_go_loss(
        relax_residuals=td_residuals,
        td_gamma=0.8,
        gae_lambda=0.99,
    )

    assert loss.shape == ()


def test_relax_to_go_loss_batch(loss_and_grad: LossAndGrad):
    horizon = 20
    batch_size = 200
    key = jax.random.key(42)
    td_residuals = jax.random.normal(key, (batch_size, batch_size, horizon - 1))

    loss = loss_and_grad.relax_to_go_loss(
        relax_residuals=td_residuals,
        td_gamma=0.8,
        gae_lambda=0.99,
    )

    assert loss.shape == ()


def test_value_residuals_single(
    loss_and_grad: LossAndGrad, value_to_go: DropoutMLP, state_dim, action_dim
):
    horizon = 20
    key = jax.random.key(42)
    keys = jax.random.split(key, 2)
    states = jax.random.normal(keys[0], (horizon, state_dim))
    actions = jax.random.normal(keys[1], (horizon, action_dim))

    value_residuals, _ = loss_and_grad.value_residuals(
        states=states,
        actions=actions,
        value_to_go_state=nnx.state(value_to_go),
        td_gamma=0.8,
    )

    assert value_residuals.shape == (horizon - 1,)


def test_value_residuals_batch(
    loss_and_grad: LossAndGrad, value_to_go: DropoutMLP, state_dim, action_dim
):
    horizon = 20
    batch_size = 200
    key = jax.random.key(42)
    keys = jax.random.split(key, 2)
    states = jax.random.normal(
        keys[0], (batch_size, batch_size, horizon, state_dim)
    )
    actions = jax.random.normal(
        keys[1], (batch_size, batch_size, horizon, action_dim)
    )

    value_residuals, _ = loss_and_grad.value_residuals(
        states=states,
        actions=actions,
        value_to_go_state=nnx.state(value_to_go),
        td_gamma=0.8,
    )

    assert value_residuals.shape == (
        batch_size,
        batch_size,
        horizon - 1,
    )


def test_relax_residuals_single(
    loss_and_grad: LossAndGrad,
    relax_to_go: DropoutMLP,
    state_dim,
):
    horizon = 20
    constraint_dim = loss_and_grad.projection.sys_constraints.constraint_dim
    key = jax.random.key(42)
    keys = jax.random.split(key, 3)
    states = jax.random.normal(keys[0], (horizon, state_dim))
    relaxations_plus = jax.random.normal(keys[1], (horizon, constraint_dim))
    relaxations_minus = jax.random.normal(keys[2], (horizon, constraint_dim))

    value_residuals, _ = loss_and_grad.relax_residuals(
        states=states,
        relaxations_plus=relaxations_plus,
        relaxations_minus=relaxations_minus,
        relax_to_go_state=nnx.state(relax_to_go),
        td_gamma=0.8,
    )

    assert value_residuals.shape == (horizon - 1,)


def test_relax_residuals_batch(
    loss_and_grad: LossAndGrad,
    relax_to_go: DropoutMLP,
    state_dim,
):
    horizon = 20
    batch_size = 200
    constraint_dim = loss_and_grad.projection.sys_constraints.constraint_dim
    key = jax.random.key(42)
    keys = jax.random.split(key, 3)
    states = jax.random.normal(
        keys[0], (batch_size, batch_size, horizon, state_dim)
    )
    relaxations_plus = jax.random.normal(
        keys[1], (batch_size, batch_size, horizon, constraint_dim)
    )
    relaxations_minus = jax.random.normal(
        keys[2], (batch_size, batch_size, horizon, constraint_dim)
    )

    value_residuals, _ = loss_and_grad.relax_residuals(
        states=states,
        relaxations_plus=relaxations_plus,
        relaxations_minus=relaxations_minus,
        relax_to_go_state=nnx.state(relax_to_go),
        td_gamma=0.8,
    )

    assert value_residuals.shape == (
        batch_size,
        batch_size,
        horizon - 1,
    )


def test_rollout(  # noqa: PLR0913 test clarity
    loss_and_grad: LossAndGrad,
    policy: PolicyBase,
    relax_to_go: DropoutMLP,
    state_dim,
    action_dim,
    horizon,
):
    constraint_dim = loss_and_grad.projection.sys_constraints.constraint_dim
    batch_size = 512
    key = jax.random.key(42)
    keys = jax.random.split(key, 2)
    initial_states = jax.random.normal(keys[0], (batch_size, state_dim))
    jax_keys = jax.random.split(keys[1], batch_size)

    rollout = loss_and_grad.calculate_rollout(
        policy_state=nnx.state(policy),
        relax_to_go_state=nnx.state(relax_to_go),
        initial_states=initial_states,
        jax_keys=jax_keys,
    )

    assert rollout.states.shape == (
        batch_size,
        horizon,
        state_dim,
    )
    assert rollout.actions.shape == (
        batch_size,
        horizon,
        action_dim,
    )
    assert rollout.interventions.shape == (
        batch_size,
        horizon,
        action_dim,
    )
    assert rollout.relaxations_plus.shape == (
        batch_size,
        horizon,
        constraint_dim,
    )
    assert rollout.relaxations_minus.shape == (
        batch_size,
        horizon,
        constraint_dim,
    )


def test_call(
    loss_and_grad: LossAndGrad,
    policy: PolicyBase,
    relax_to_go: DropoutMLP,
    value_to_go: DropoutMLP,
    state_dim,
):
    batch_size = 512
    key = jax.random.key(42)
    keys = jax.random.split(key, 2)
    initial_states = jax.random.normal(keys[0], (batch_size, state_dim))
    jax_keys = jax.random.split(keys[1], batch_size)

    (a, b, c) = loss_and_grad(
        policy_state=nnx.state(policy),
        relax_to_go_state=nnx.state(relax_to_go),
        value_to_go_state=nnx.state(value_to_go),
        initial_states=initial_states,
        jax_keys=jax_keys,
        td_gamma=0.8,
        gae_lambda=0.99,
    )
