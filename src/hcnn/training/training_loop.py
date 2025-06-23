import os

import jax
import jax.numpy as jnp
import optax
from flax import nnx

import wandb
from hcnn.optimization import OptimizationSolverBase, OptimizationSolverShallow
from hcnn.policy import PolicyBase, PolicyMLP
from hcnn.projection import ProjectionBase, ProjectionNoSubsNoTerm
from hcnn.system_constraints import SystemConstraintsBase, SystemConstraintsBox
from hcnn.system_costs import SystemCostsBase, SystemCostsQuadratic
from hcnn.system_dynamics import SystemDynamicsBase, SystemDynamicsLinear
from hcnn.training.dropout_mlp import DropoutMLP
from hcnn.training.loss_and_grad import LossAndGrad

activation_map = {
    "celu": nnx.celu,
    "relu": nnx.relu,
    "tanh": nnx.tanh,
}


def build_system_dynamics(config) -> SystemDynamicsBase:
    if config.system_name == "double_integrator":
        return SystemDynamicsLinear.factory(
            state_matrix=jnp.array([[0, 1], [0, 0]]),
            action_matrix=jnp.array([[0], [1]]),
        )
    else:
        raise ValueError(f"Unknown system: {config.system_name}")


def build_system_costs(config) -> SystemCostsBase:
    if config.system_name == "double_integrator":
        return SystemCostsQuadratic.factory(
            state_cost_matrix=jnp.array([[1, 0], [0, 1]]),
            action_cost_matrix=jnp.array([[1]]),
        )
    else:
        raise ValueError(f"Unknown system: {config.system_name}")


def build_system_constraints(config) -> SystemConstraintsBase:
    if config.system_name == "double_integrator":
        return SystemConstraintsBox.factory(
            state_l=jnp.array([-1, -1]),
            state_u=jnp.array([1, 1]),
            action_l=jnp.array([-1]),
            action_u=jnp.array([1]),
        )
    else:
        raise ValueError(f"Unknown system: {config.system_name}")


def build_policy(config, state_dim, action_dim, rngs) -> PolicyBase:
    if config.policy_name == "dropout_mlp":
        return PolicyMLP.factory(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layer_sizes=[config.policy_hidden_dim]
            * config.policy_hidden_depth,
            dropout_rate=config.policy_dropout_rate,
            activation=activation_map[config.policy_activation],
            rngs=rngs,
        )
    else:
        raise ValueError(f"Unknown system: {config.system_name}")


def build_cost_to_go(config, state_dim, rngs) -> DropoutMLP:
    return DropoutMLP(
        layer_sizes=[state_dim]
        + [config.cost_to_go_hidden_dim] * config.cost_to_go_hidden_depth
        + [1],
        dropout_rate=config.cost_to_go_dropout_rate,
        activation=activation_map[config.cost_to_go_activation],
        rngs=rngs,
    )


def build_relax_to_go(config, state_dim, rngs) -> DropoutMLP:
    return DropoutMLP(
        layer_sizes=[state_dim]
        + [config.relax_to_go_hidden_dim] * config.relax_to_go_hidden_depth
        + [1],
        dropout_rate=config.relax_to_go_dropout_rate,
        activation=activation_map[config.relax_to_go_activation],
        rngs=rngs,
    )


def build_projection_operator(
    config,
    policy,
    relax_to_go,
    sys_dynamics,
    sys_constraints,
) -> ProjectionBase:
    if config.projection_name == "no_substitution_no_terminal":
        return ProjectionNoSubsNoTerm.factory(
            policy=policy,
            relax_to_go=relax_to_go,
            sys_dynamics=sys_dynamics,
            sys_constraints=sys_constraints,
            horizon=config.projection_horizon,
            relaxation_penalty=config.projection_relaxation_penalty,
        )
    else:
        raise ValueError(f"Unknown system: {config.system_name}")


def build_solver(config) -> OptimizationSolverBase:
    if config.solver_name == "shallow":
        return OptimizationSolverShallow(
            max_iter=config.solver_max_iter,
            tol=config.solver_tol,
        )
    else:
        raise ValueError(f"Unknown system: {config.system_name}")


def build_loss_and_grad(
    projection, solver, sys_costs, cost_to_go, relax_to_go
) -> LossAndGrad:
    return LossAndGrad.factory(
        projection=projection,
        solver=solver,
        sys_costs=sys_costs,
        cost_to_go=cost_to_go,
        relax_to_go=relax_to_go,
    )


def build_optimizer(model, name, learning_rate):
    opt_map = {
        "adabelief": optax.adabelief,
        "nadamw": optax.nadamw,
    }

    if name in opt_map:
        return nnx.Optimizer(model, opt_map[name](learning_rate=learning_rate))
    else:
        raise ValueError(f"Unknown optimizer: {name}")


class TrainBundle(nnx.Module):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# @jax.jit
def train_epoch(
    config,
    train_graphdef,
    train_state,
):
    # Calculate the number of batches per epoch
    batch_per_epoch = config.samples_per_epoch // config.batchsize

    # Hydrate the train state with the graph definition
    train = nnx.merge(train_graphdef, train_state)

    def train_step(train):
        # sample initial states and JAX keys
        # TODO replace by sampler
        initial_states = jax.random.normal(train.rngs.default(), (8, 2))
        jax_keys = jax.random.split(train.rngs.default(), (8,))

        loss_tuple, grad_tuple, new_states = train.loss_and_grad(
            policy_state=nnx.state(train.policy),
            relax_to_go_state=nnx.state(train.relax_to_go),
            cost_to_go_state=nnx.state(train.cost_to_go),
            initial_states=initial_states,
            jax_keys=jax_keys,
            td_gamma=0.9,
            gae_lambda=0.99,
        )
        # Update metrics
        train.metrics.update(
            policy_loss=loss_tuple[0],
            cost_to_go_loss=loss_tuple[1],
            relax_to_go_loss=loss_tuple[2],
            mean_loss=jnp.mean(loss_tuple),
        )

        # Update the states of the models
        train.opt_policy.update(grad_tuple[0])
        train.opt_cost.update(grad_tuple[1])
        train.opt_relax.update(grad_tuple[2])

        # Update model states
        nnx.update(train.policy, new_states[0])
        nnx.update(train.cost, new_states[1])
        nnx.update(train.policy, new_states[2])

    train_step(train)


def train(project=None, config=None):
    with wandb.init(project=project, config=config):
        # If called by wandb.agent,
        # this config will be set by Sweep Controller
        config = wandb.config

        # Build system model components
        sys_dynamics = build_system_dynamics(config)
        sys_costs = build_system_costs(config)
        sys_constraints = build_system_constraints(config)

        # Check consistency of dimensions
        state_dim = sys_dynamics.state_dim
        action_dim = sys_dynamics.action_dim
        assert state_dim == sys_costs.state_dim
        assert action_dim == sys_costs.action_dim
        assert state_dim == sys_constraints.state_dim
        assert action_dim == sys_constraints.action_dim

        rngs = nnx.Rngs(config.seed)

        # Build the policy
        policy = build_policy(config, state_dim, action_dim, rngs)

        # Build the "to-go" estimators
        cost_to_go = build_cost_to_go(config, state_dim, rngs)
        relax_to_go = build_relax_to_go(config, state_dim, rngs)

        # Build the projection operator
        projection = build_projection_operator(
            config,
            policy=policy,
            relax_to_go=relax_to_go,
            sys_dynamics=sys_dynamics,
            sys_constraints=sys_constraints,
        )

        # Build the solver
        solver = build_solver(config)

        # Build the loss and gradient computation
        loss_and_grad = build_loss_and_grad(
            projection=projection,
            solver=solver,
            sys_costs=sys_costs,
            cost_to_go=cost_to_go,
            relax_to_go=relax_to_go,
        )

        # Build optimizers
        opt_policy = build_optimizer(
            policy,
            config.optimizer_policy_name,
            config.optimizer_policy_learning_rate,
        )
        opt_cost = build_optimizer(
            cost_to_go,
            config.optimizer_cost_name,
            config.optimizer_cost_learning_rate,
        )
        opt_relax = build_optimizer(
            relax_to_go,
            config.optimizer_relax_name,
            config.optimizer_relax_learning_rate,
        )

        # Build optax metrics
        metrics = nnx.MultiMetric(
            policy_loss=nnx.metrics.Average("policy_loss"),
            cost_to_go_loss=nnx.metrics.Average("cost_to_go_loss"),
            relax_to_go_loss=nnx.metrics.Average("relax_to_go_loss"),
            mean_loss=nnx.metrics.Average("mean_loss"),
        )

        train = TrainBundle(
            policy=policy,
            cost_to_go=cost_to_go,
            relax_to_go=relax_to_go,
            loss_and_grad=loss_and_grad,
            opt_policy=opt_policy,
            opt_cost=opt_cost,
            opt_relax=opt_relax,
            metrics=metrics,
            rngs=rngs,
        )

        train_graphdef, train_state = nnx.split(train)

        train_epoch(config, train_graphdef, train_state)

        wandb.log({"test_out": 42})


if __name__ == "__main__":
    print("This is a training loop with Weights & Biases integration.")

    # restrict access to one GPU (check availability nvidia-smi)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # Specify the project name and configuration
    # This can be overridden by wandb.agent when running a sweep
    project = "double_integrator"
    config = {
        "system_name": "double_integrator",
        "seed": 42,
        "policy_name": "dropout_mlp",
        "policy_hidden_dim": 8,
        "policy_hidden_depth": 4,
        "policy_dropout_rate": 0.2,
        "policy_activation": "celu",
        "cost_to_go_hidden_dim": 8,
        "cost_to_go_hidden_depth": 4,
        "cost_to_go_dropout_rate": 0.2,
        "cost_to_go_activation": "celu",
        "relax_to_go_hidden_dim": 8,
        "relax_to_go_hidden_depth": 4,
        "relax_to_go_dropout_rate": 0.2,
        "relax_to_go_activation": "celu",
        "projection_name": "no_substitution_no_terminal",
        "projection_horizon": 20,
        "projection_relaxation_penalty": 7.0,
        "solver_name": "shallow",
        "solver_max_iter": 500,
        "solver_tol": 0.01,
        "optimizer_policy_name": "nadamw",
        "optimizer_policy_learning_rate": 1e-3,
        "optimizer_cost_name": "nadamw",
        "optimizer_cost_learning_rate": 1e-2,
        "optimizer_relax_name": "nadamw",
        "optimizer_relax_learning_rate": 1e-2,
        "samples_per_epoch": 16,
        "batchsize": 1,
        "td_gamma": 0.9,
        "gae_lambda": 0.99,
    }

    # Start the training loop
    train(project=project, config=config)
