import jax
import pytest
from flax import nnx

from hcnn.training.dropout_mlp import DropoutMLP
from hcnn.training.loss_and_grad import LossAndGrad
from tests.projection.test_projection_no_subs_no_term import (
    projection,  # noqa: F401
)
from tests.system_cost import sys_cost


@pytest.fixture
def value_to_go():
    return DropoutMLP(
        layer_sizes=(64, 1),
        dropout_rate=0.1,
        rngs=nnx.Rngs(42),
    )


@pytest.fixture
def relax_to_go():
    return DropoutMLP(
        layer_sizes=(64, 1),
        dropout_rate=0.1,
        rngs=nnx.Rngs(42),
    )


@pytest.fixture
def loss_and_grad(
    projection,
    value_to_go,
    relax_to_go,
) -> LossAndGrad:
    return LossAndGrad.factory(
        projection=projection,
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
    )

    assert loss.shape == ()


def test_policy_loss_batch(loss_and_grad: LossAndGrad):
    batch_size = 200
    horizon = 20
    key = jax.random.key(42)
    value_residuals = jax.random.normal(key, (20, batch_size, horizon - 1))
    rollout_interventions = jax.random.normal(20, key, (batch_size, horizon))

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
    )

    assert loss.shape == ()


def test_value_to_go_loss_batch(loss_and_grad: LossAndGrad):
    horizon = 20
    batch_size = 200
    key = jax.random.key(42)
    td_residuals = jax.random.normal(key, (batch_size, horizon - 1))

    loss = loss_and_grad.value_to_go_loss(
        value_residuals=td_residuals,
    )

    assert loss.shape == ()


def test_relax_to_go_loss_single(loss_and_grad: LossAndGrad):
    horizon = 20
    key = jax.random.key(42)
    td_residuals = jax.random.normal(key, (horizon - 1,))

    loss = loss_and_grad.relax_to_go_loss(
        relax_residuals=td_residuals,
    )

    assert loss.shape == ()


def test_relax_to_go_loss_batch(loss_and_grad: LossAndGrad):
    horizon = 20
    batch_size = 200
    key = jax.random.key(42)
    td_residuals = jax.random.normal(key, (batch_size, horizon - 1))

    loss = loss_and_grad.relax_to_go_loss(
        relax_residuals=td_residuals,
    )

    assert loss.shape == ()


def test_value_residuals(sys_cost=sys_cost):
    pass
