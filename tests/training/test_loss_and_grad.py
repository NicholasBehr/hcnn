import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hcnn.optimization import OptimizationSolverShallow
from hcnn.policy import PolicyBase
from hcnn.system_constraints import SystemConstraintsBase
from hcnn.system_costs import SystemCostsBase
from hcnn.system_dynamics import SystemDynamicsBase
from hcnn.training.dropout_mlp import DropoutMLP
from hcnn.training.loss_and_grad import LossAndGrad
from tests.policy import PolicyMock
from tests.projection import ProjectionMock
from tests.system_constraints import SystemConstraintsMock
from tests.system_costs import SystemCostMock
from tests.system_dynamics import SystemDynamicsMock


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
def cost_to_go(state_dim):
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
def sys_costs(state_dim, action_dim) -> SystemCostsBase:
    return SystemCostMock(
        state_dim=state_dim,
        action_dim=action_dim,
    )


@pytest.fixture
def solver():
    return OptimizationSolverShallow(tol=0.01)


@pytest.fixture
def loss_and_grad(
    projection,
    solver,
    sys_costs,
    cost_to_go,
    relax_to_go,
) -> LossAndGrad:
    return LossAndGrad.factory(
        projection=projection,
        sys_costs=sys_costs,
        solver=solver,
        cost_to_go=cost_to_go,
        relax_to_go=relax_to_go,
    )


def test_policy_loss_single(loss_and_grad: LossAndGrad):
    horizon = 20
    key = jax.random.key(42)
    value_residuals = jax.random.normal(key, (horizon - 1,))
    rollout_interventions = jax.random.normal(key, (horizon,))

    loss = loss_and_grad.policy_loss(
        cost_residuals=value_residuals,
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
        cost_residuals=value_residuals,
        rollout_interventions=rollout_interventions,
        td_gamma=0.8,
        gae_lambda=0.99,
    )

    assert loss.shape == ()


def test_value_to_go_loss_single(loss_and_grad: LossAndGrad):
    horizon = 20
    key = jax.random.key(42)
    td_residuals = jax.random.normal(key, (horizon - 1,))

    loss = loss_and_grad.cost_to_go_loss(
        cost_residuals=td_residuals,
        td_gamma=0.8,
        gae_lambda=0.99,
    )

    assert loss.shape == ()


def test_value_to_go_loss_batch(loss_and_grad: LossAndGrad):
    horizon = 20
    batch_size = 200
    key = jax.random.key(42)
    td_residuals = jax.random.normal(key, (batch_size, batch_size, horizon - 1))

    loss = loss_and_grad.cost_to_go_loss(
        cost_residuals=td_residuals,
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
    loss_and_grad: LossAndGrad, cost_to_go: DropoutMLP, state_dim, action_dim
):
    horizon = 20
    key = jax.random.key(42)
    keys = jax.random.split(key, 2)
    states = jax.random.normal(keys[0], (horizon, state_dim))
    actions = jax.random.normal(keys[1], (horizon, action_dim))

    value_residuals, _ = loss_and_grad.cost_residuals(
        states=states,
        actions=actions,
        cost_to_go_state=nnx.state(cost_to_go),
        td_gamma=0.8,
    )

    assert value_residuals.shape == (horizon - 1,)


def test_value_residuals_batch(
    loss_and_grad: LossAndGrad, cost_to_go: DropoutMLP, state_dim, action_dim
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

    value_residuals, _ = loss_and_grad.cost_residuals(
        states=states,
        actions=actions,
        cost_to_go_state=nnx.state(cost_to_go),
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


def test_rollout_fwd(  # noqa: PLR0913 test clarity
    loss_and_grad: LossAndGrad,
    policy: PolicyBase,
    relax_to_go: DropoutMLP,
    state_dim,
    action_dim,
    horizon,
):
    constraint_dim = loss_and_grad.projection.sys_constraints.constraint_dim
    batch_size = 32
    key = jax.random.key(42)
    keys = jax.random.split(key, 2)
    initial_states = jax.random.normal(keys[0], (batch_size, state_dim))
    jax_keys = jax.random.split(keys[1], batch_size)

    policy_state = nnx.state(policy)
    relax_to_go_state = nnx.state(relax_to_go)

    rollout = loss_and_grad.calculate_rollout(
        policy_params=nnx.filter_state(policy_state, nnx.Param),
        relax_to_go_params=nnx.filter_state(relax_to_go_state, nnx.Param),
        policy_non_params=nnx.filter_state(policy_state, nnx.Not(nnx.Param)),
        relax_to_go_non_params=nnx.filter_state(
            relax_to_go_state, nnx.Not(nnx.Param)
        ),
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


def test_rollout_bwd(
    loss_and_grad: LossAndGrad,
    policy: PolicyBase,
    relax_to_go: DropoutMLP,
    state_dim,
):
    batch_size = 32
    key = jax.random.key(42)
    keys = jax.random.split(key, 2)
    initial_states = jax.random.normal(keys[0], (batch_size, state_dim))
    jax_keys = jax.random.split(keys[1], batch_size)

    policy_state = nnx.state(policy)
    relax_to_go_state = nnx.state(relax_to_go)

    rollout, rollout_vjp = jax.vjp(
        lambda policy_params,
        relax_to_go_params: loss_and_grad.calculate_rollout(
            policy_params=policy_params,
            relax_to_go_params=relax_to_go_params,
            policy_non_params=nnx.filter_state(
                policy_state, nnx.Not(nnx.Param)
            ),
            relax_to_go_non_params=nnx.filter_state(
                relax_to_go_state, nnx.Not(nnx.Param)
            ),
            initial_states=initial_states,
            jax_keys=jax_keys,
        ),
        nnx.filter_state(policy_state, nnx.Param),
        nnx.filter_state(relax_to_go_state, nnx.Param),
    )

    rollout_grad = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), rollout)

    out = rollout_vjp(rollout_grad)


def test_call_fwd(
    loss_and_grad: LossAndGrad,
    policy: PolicyBase,
    relax_to_go: DropoutMLP,
    cost_to_go: DropoutMLP,
    state_dim,
):
    batch_size = 32
    key = jax.random.key(42)
    keys = jax.random.split(key, 2)
    initial_states = jax.random.normal(keys[0], (batch_size, state_dim))
    jax_keys = jax.random.split(keys[1], batch_size)

    (a, b, c) = loss_and_grad(
        policy_state=nnx.state(policy),
        relax_to_go_state=nnx.state(relax_to_go),
        cost_to_go_state=nnx.state(cost_to_go),
        initial_states=initial_states,
        jax_keys=jax_keys,
        td_gamma=0.8,
        gae_lambda=0.99,
    )
