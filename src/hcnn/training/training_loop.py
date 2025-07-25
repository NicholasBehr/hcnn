import os
from functools import partial

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx

import wandb
from hcnn.optimization import (
    OptimizationSolverBase,
    OptimizationSolverOSQP,
    OptimizationSolverShallow,
)
from hcnn.policy import PolicyBase, PolicyMLP
from hcnn.projection import ProjectionBase, ProjectionNoSubsNoTerm
from hcnn.system_constraints import SystemConstraintsBox
from hcnn.system_costs import SystemCostsQuadratic
from hcnn.system_dynamics import (
    SystemDynamicsDoublePendulum,
    SystemDynamicsLinear,
)
from hcnn.training.dropout_mlp import DropoutMLP
from hcnn.training.loss_and_grad import LossAndGrad, ModelStates


def build_double_integrator(config):
    sys_dynamics = SystemDynamicsLinear.factory(
        state_matrix=jnp.array([[1, 1], [0, 1]]),
        action_matrix=jnp.array([[0.5], [1]]),
    )
    sys_costs = SystemCostsQuadratic.factory(
        state_cost_matrix=jnp.array([[5, 0], [0, 5]]),
        action_cost_matrix=jnp.array([[1]]),
    )
    action_range = (-1, +1)
    sys_constraints = SystemConstraintsBox.factory(
        state_l=jnp.array([-10, -10]),
        state_u=jnp.array([+10, +10]),
        action_l=jnp.array(action_range[0]),
        action_u=jnp.array(action_range[1]),
    )
    rngs = nnx.Rngs(config.seed)
    policy = PolicyMLP.factory(
        state_dim=2,
        action_dim=1,
        hidden_features=[config.policy_hidden_dim] * config.policy_hidden_depth,
        rngs=rngs,
        activation=config.policy_activation,
        dropout_rate=config.policy_dropout_rate,
        action_range=action_range,
    )
    cost_to_go = DropoutMLP(
        in_features=2,
        out_features=1,
        hidden_features=[config.cost_to_go_hidden_dim]
        * config.cost_to_go_hidden_depth,
        rngs=rngs,
        activation=config.cost_to_go_activation,
        dropout_rate=config.cost_to_go_dropout_rate,
    )
    relax_to_go = DropoutMLP(
        in_features=2,
        out_features=1,
        hidden_features=[config.relax_to_go_hidden_dim]
        * config.relax_to_go_hidden_depth,
        rngs=rngs,
        activation=config.relax_to_go_activation,
        dropout_rate=config.relax_to_go_dropout_rate,
    )
    return (
        sys_dynamics,
        sys_costs,
        sys_constraints,
        rngs,
        policy,
        cost_to_go,
        relax_to_go,
    )


def build_double_pendulum(config):
    sys_dynamics = SystemDynamicsDoublePendulum.factory(dt=0.02)
    sys_costs = SystemCostsQuadratic.factory(
        state_cost_matrix=jnp.diag(jnp.array([1.0, 1.0, 0.1, 0.1])),
        action_cost_matrix=jnp.diag(jnp.array([0.1])),
        reference_state=jnp.array([jnp.pi, 0.0, 0.0, 0.0]),
        reference_action=jnp.array([0.0]),
        periodic_states=frozenset({0, 1}),  # Pendulum angles are periodic
    )
    action_range = (-1, +1)
    sys_constraints = SystemConstraintsBox.factory(
        state_l=jnp.array([-jnp.inf, -jnp.inf, -3, -3]),
        state_u=jnp.array([+jnp.inf, +jnp.inf, +3, +3]),
        action_l=action_range[0] * jnp.ones(sys_dynamics.action_dim),
        action_u=action_range[1] * jnp.ones(sys_dynamics.action_dim),
    )
    rngs = nnx.Rngs(config.seed)
    periodic_states = frozenset({0, 1})  # Pendulum angles are periodic
    policy = PolicyMLP.factory(
        state_dim=4,
        action_dim=1,
        hidden_features=[config.policy_hidden_dim] * config.policy_hidden_depth,
        rngs=rngs,
        activation=config.policy_activation,
        dropout_rate=config.policy_dropout_rate,
        periodic_states=periodic_states,
        action_range=action_range,
    )
    cost_to_go = DropoutMLP(
        in_features=4,
        out_features=1,
        hidden_features=[config.cost_to_go_hidden_dim]
        * config.cost_to_go_hidden_depth,
        rngs=rngs,
        activation=config.cost_to_go_activation,
        dropout_rate=config.cost_to_go_dropout_rate,
        periodic_in_features=periodic_states,
    )
    relax_to_go = DropoutMLP(
        in_features=4,
        out_features=1,
        hidden_features=[config.relax_to_go_hidden_dim]
        * config.relax_to_go_hidden_depth,
        rngs=rngs,
        activation=config.relax_to_go_activation,
        dropout_rate=config.relax_to_go_dropout_rate,
        periodic_in_features=periodic_states,
    )
    return (
        sys_dynamics,
        sys_costs,
        sys_constraints,
        rngs,
        policy,
        cost_to_go,
        relax_to_go,
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
    if config.solver_name == "osqp":
        solver = OptimizationSolverOSQP
    elif config.solver_name == "shallow":
        solver = OptimizationSolverShallow
    else:
        raise ValueError(f"Unknown system: {config.system_name}")

    return solver(
        alpha=config.solver_alpha,
        sigma=config.solver_sigma,
        rho_equality=config.solver_rho_equality,
        rho_inequality=config.solver_rho_inequality,
        max_iter=config.solver_max_iter,
        tol=config.solver_tol,
        gradient_clip=config.solver_gradient_clip,
    )


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


def build_optimizer(model, name: str, learning_rate):
    optimizer_map = {
        "adabelief": optax.adabelief,
        "adagrad": optax.adagrad,
        "adam": optax.adam,
        "adamw": optax.adamw,
        "nadamw": optax.nadamw,
    }
    return nnx.Optimizer(
        model, optimizer_map[name](learning_rate=learning_rate)
    )


class TrainBundle(nnx.Module):
    policy: PolicyBase
    cost_to_go: DropoutMLP
    relax_to_go: DropoutMLP
    loss_and_grad: LossAndGrad
    opt_policy: nnx.Optimizer
    opt_cost: nnx.Optimizer
    opt_relax: nnx.Optimizer
    metrics: nnx.MultiMetric
    rngs: nnx.Rngs

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


@partial(
    jax.jit,
    static_argnames=["config", "train_graphdef"],
    donate_argnums=(2,),  # Donate train_state to reduce memory overhead
)
def train_epoch(
    config,
    train_graphdef,
    train_state,
):
    # Pre-compile scalar constants to avoid recompilation
    batchsize = config.batchsize
    td_gamma = config.td_gamma
    gae_lambda = config.gae_lambda
    intervention_cost = config.policy_intervention_cost

    def train_step(step_idx, train_state):
        train = nnx.merge(train_graphdef, train_state)

        # Use step-dependent but deterministic sampling to reduce
        # compilation variance
        step_key = jax.random.fold_in(train.rngs.default(), step_idx)
        sample_key, split_key = jax.random.split(step_key)

        initial_states = jax.random.uniform(
            sample_key,
            (batchsize, train.policy.state_dim),
            minval=-2 * jnp.pi,
            maxval=+2 * jnp.pi,
        )
        jax_keys = jax.random.split(split_key, batchsize)

        model_states = ModelStates(
            policy=nnx.state(train.policy),
            relax_to_go=nnx.state(train.relax_to_go),
            cost_to_go=nnx.state(train.cost_to_go),
        )

        # Use cached function parameters to reduce recompilation
        loss, grad, model_states, info = train.loss_and_grad(
            model_states=model_states,
            initial_states=initial_states,
            jax_keys=jax_keys,
            td_gamma=td_gamma,
            gae_lambda=gae_lambda,
            intervention_cost=intervention_cost,
        )

        # Update model states with more efficient operations
        nnx.update(train.policy, model_states.policy)
        nnx.update(train.relax_to_go, model_states.relax_to_go)
        nnx.update(train.cost_to_go, model_states.cost_to_go)

        # Optimize aggregate loss computation
        loss_array = jnp.array([loss.policy, loss.cost_to_go, loss.relax_to_go])
        aggregate_loss = jnp.sum(jnp.log1p(jnp.abs(loss_array)))

        # Compute maximum absolute gradient value for monitoring and clipping
        grad_leaves = jax.tree_util.tree_leaves(grad.policy)
        max_gradient = jnp.max(
            jnp.array([jnp.max(jnp.abs(leaf)) for leaf in grad_leaves])
        )
        mean_gradient = jnp.mean(
            jnp.array([jnp.mean(jnp.abs(leaf)) for leaf in grad_leaves])
        )

        # Update metrics with batched operations
        train.metrics.update(
            policy_loss=loss.policy,
            cost_to_go_loss=loss.cost_to_go,
            relax_to_go_loss=loss.relax_to_go,
            aggregate_loss=aggregate_loss,
            primal_residual=info.primal_residual,
            dual_residual=info.dual_residual,
            solver_iterations=info.solver_iterations,
            max_gradient=max_gradient,
            mean_gradient=mean_gradient,
        )

        # Update optimizers with pre-allocated gradients
        train.opt_policy.update(grad.policy)
        train.opt_relax.update(grad.relax_to_go)
        train.opt_cost.update(grad.cost_to_go)

        return nnx.state(train)

    return jax.lax.fori_loop(
        lower=0,
        upper=config.batches_per_epoch,
        body_fun=train_step,
        init_val=train_state,
    )


def train(project=None, config=None):  # noqa: PLR0915 code is using wandb reference design
    with wandb.init(project=project, config=config):
        # If called by wandb.agent,
        # this config will be set by Sweep Controller
        config = wandb.config

        system_builders = {
            "double_integrator": build_double_integrator,
            "double_pendulum": build_double_pendulum,
        }
        builder = system_builders.get(config.system_name)

        if not builder:
            raise ValueError(f"Unknown system: {config.system_name}")

        (
            sys_dynamics,
            sys_costs,
            sys_constraints,
            rngs,
            policy,
            cost_to_go,
            relax_to_go,
        ) = builder(config)

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
            # model evaluation metrics
            policy_loss=nnx.metrics.Average("policy_loss"),
            cost_to_go_loss=nnx.metrics.Average("cost_to_go_loss"),
            relax_to_go_loss=nnx.metrics.Average("relax_to_go_loss"),
            aggregate_loss=nnx.metrics.Average("aggregate_loss"),
            # solver metrics
            primal_residual=nnx.metrics.Average("primal_residual"),
            dual_residual=nnx.metrics.Average("dual_residual"),
            solver_iterations=nnx.metrics.Average("solver_iterations"),
            # training metrics
            max_gradient=nnx.metrics.Average("max_gradient"),
            mean_gradient=nnx.metrics.Average("mean_gradient"),
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

        for epoch in range(config.epochs):
            train_state = train_epoch(
                config=config,
                train_graphdef=train_graphdef,
                train_state=train_state,
            )

            train = nnx.merge(train_graphdef, train_state)
            metric_dict = train.metrics.compute()
            train.metrics.reset()
            train_state = nnx.state(train)

            wandb.log(step=epoch, data=metric_dict)

            # Print metrics in a nicely formatted table
            if epoch == 0:
                header = f"{'Epoch':<8}" + "".join(
                    [f"{k:<20}" for k in metric_dict.keys()]
                )
                print(header)
            values = f"{epoch:<8}" + "".join(
                [f"{float(v):<20.6f}" for v in metric_dict.values()]
            )
            print(values)

            if epoch % 10 == 0 or epoch == config.epochs - 1:
                # Plot the model
                samples_per_axis = 300
                axis_samples = jnp.linspace(
                    -2 * jnp.pi,
                    +2 * jnp.pi,
                    samples_per_axis,
                )
                x_coord, y_coord = jnp.meshgrid(
                    axis_samples,
                    axis_samples,
                    indexing="xy",
                )

                # Flatten the grid to create a list of states
                # This is necessary for batch processing in JAX
                # We stack the coordinates to create a 2D array of states
                # For 2D systems, use (x, y); for 4D systems, use (θ₁, θ₂, 0, 0)
                state_dim_2d = 2
                state_dim_4d = 4
                if train.policy.state_dim == state_dim_2d:
                    states = jnp.stack(
                        (x_coord.flatten(), y_coord.flatten()), axis=-1
                    )
                elif train.policy.state_dim == state_dim_4d:
                    # For double pendulum, visualize in (θ₁, θ₂) space
                    # with zero velocities
                    states = jnp.stack(
                        (
                            x_coord.flatten(),
                            y_coord.flatten(),
                            jnp.zeros_like(x_coord.flatten()),
                            jnp.zeros_like(y_coord.flatten()),
                        ),
                        axis=-1,
                    )
                else:
                    print(
                        f"Skipping visualization for "
                        f"{train.policy.state_dim}D state space"
                    )
                    continue

                x = states[:, 0].reshape(samples_per_axis, samples_per_axis)
                y = states[:, 1].reshape(samples_per_axis, samples_per_axis)

                # Plot cost-to-go
                train.cost_to_go.eval()
                costs = (
                    train.cost_to_go(states)
                    .squeeze()
                    .reshape(samples_per_axis, samples_per_axis)
                )
                train.cost_to_go.train()

                plt.figure(figsize=(9, 7))
                plt.pcolormesh(x, y, costs, cmap="viridis", shading="gouraud")
                plt.colorbar()
                wandb.log(step=epoch, data={"costs-to-go": wandb.Image(plt)})
                plt.close()

                # Plot relax-to-go
                train.relax_to_go.eval()
                relax = (
                    train.relax_to_go(states)
                    .squeeze()
                    .reshape(samples_per_axis, samples_per_axis)
                )
                train.relax_to_go.train()

                plt.figure(figsize=(9, 7))
                plt.pcolormesh(x, y, relax, cmap="viridis", shading="gouraud")
                plt.colorbar()
                wandb.log(step=epoch, data={"relax-to-go": wandb.Image(plt)})
                plt.close()

                # Plot policy actions
                if policy.action_dim == 1:
                    train.policy.eval()
                    actions = (
                        train.policy(states)
                        .squeeze()
                        .reshape(samples_per_axis, samples_per_axis)
                    )
                    train.policy.train()

                    plt.figure(figsize=(9, 7))
                    plt.pcolormesh(
                        x, y, actions, cmap="viridis", shading="gouraud"
                    )
                    plt.colorbar()
                    wandb.log(
                        step=epoch, data={"policy-actions": wandb.Image(plt)}
                    )
                    plt.close()


if __name__ == "__main__":
    print("This is a training loop with Weights & Biases integration.")

    # restrict access to one GPU (check availability nvidia-smi)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    jax.config.update("jax_debug_nans", False)

    # jax.config.update("jax_platform_name", "cpu")

    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )

    # Specify the project name and configuration
    # This can be overridden by wandb.agent when running a sweep
    project = "double_pendulum"
    config = {
        "system_name": "double_pendulum",
        "seed": 0,
        "policy_name": "dropout_mlp",
        "policy_hidden_dim": 6,
        "policy_hidden_depth": 6,
        "policy_dropout_rate": 0.15,
        "policy_activation": "leaky_relu",
        "policy_intervention_cost": 0.5,
        "cost_to_go_hidden_dim": 6,
        "cost_to_go_hidden_depth": 6,
        "cost_to_go_dropout_rate": 0.15,
        "cost_to_go_activation": "leaky_relu",
        "relax_to_go_hidden_dim": 6,
        "relax_to_go_hidden_depth": 6,
        "relax_to_go_dropout_rate": 0.15,
        "relax_to_go_activation": "leaky_relu",
        "projection_name": "no_substitution_no_terminal",
        "projection_horizon": 15,
        "projection_relaxation_penalty": 100,
        "solver_name": "shallow",
        "solver_alpha": 0.25,
        "solver_sigma": 0.25,
        "solver_rho_equality": 1e-1,
        "solver_rho_inequality": 1e-1,
        "solver_max_iter": 1e4,
        "solver_tol": 1e-4,
        "solver_gradient_clip": 1e4,
        "optimizer_policy_name": "adamw",
        "optimizer_policy_learning_rate": 1e-3,
        "optimizer_cost_name": "adamw",
        "optimizer_cost_learning_rate": 1e-2,
        "optimizer_relax_name": "adamw",
        "optimizer_relax_learning_rate": 1e-2,
        "batchsize": 512,
        "batches_per_epoch": 5,
        "td_gamma": 0.9,
        "gae_lambda": 0.8,
        "epochs": 200,
    }

    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "solver_iterations"},
        # "early_terminate": {"type": "hyperband", "min_iter": 20, "eta": 1.5},
        "parameters": {
            "system_name": {"value": "double_pendulum"},
            "seed": {"value": 42},
            "policy_name": {"value": "dropout_mlp"},
            "policy_hidden_dim": {"value": 6},
            "policy_hidden_depth": {"value": 6},
            "policy_dropout_rate": {"value": 0.15},
            "policy_activation": {"value": "leaky_relu"},
            "policy_intervention_cost": {"value": 0.5},
            # Cost-to-go
            "cost_to_go_hidden_dim": {"value": 6},
            "cost_to_go_hidden_depth": {"value": 6},
            "cost_to_go_dropout_rate": {"value": 0.15},
            "cost_to_go_activation": {"value": "leaky_relu"},
            # Relax-to-go
            "relax_to_go_hidden_dim": {"value": 6},
            "relax_to_go_hidden_depth": {"value": 6},
            "relax_to_go_dropout_rate": {"value": 0.15},
            "relax_to_go_activation": {"value": "leaky_relu"},
            "projection_name": {"value": "no_substitution_no_terminal"},
            "projection_horizon": {"value": 50},
            "projection_relaxation_penalty": {"value": 100},
            "solver_name": {"value": "shallow"},
            "solver_alpha": {"value": 0.1},
            "solver_sigma": {
                "distribution": "log_uniform_values",
                "max": 1.0,
                "min": 0.01,
            },
            "solver_rho_equality": {
                "distribution": "log_uniform_values",
                "max": 1e-1,
                "min": 1e-4,
            },
            "solver_rho_inequality": {
                "distribution": "log_uniform_values",
                "max": 1e-1,
                "min": 1e-4,
            },
            "solver_max_iter": {"value": 1e3},
            "solver_tol": {"value": 1e-1},
            "optimizer_policy_name": {"value": "adamw"},
            "optimizer_policy_learning_rate": {"value": 1e-2},
            "optimizer_cost_name": {"value": "adamw"},
            "optimizer_cost_learning_rate": {"value": 1e-2},
            "optimizer_relax_name": {"value": "adamw"},
            "optimizer_relax_learning_rate": {"value": 1e-2},
            "batchsize": {"value": 128},
            "batches_per_epoch": {"value": 1},
            "td_gamma": {"value": 0.95},
            "gae_lambda": {"value": 0.99},
            "epochs": {"value": 1},
        },
    }

    train(project=project, config=config)

    # sweep_id = None  # Replace with your sweep ID if you have one
    # if sweep_id is None:
    #     print("No sweep ID provided, creating sweep.")
    #     sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)
    #     print(f"Sweep ID: {sweep_id}")
    # wandb.agent(
    #     sweep_id=sweep_id,
    #     function=train,
    #     project=project,
    #     count=50,
    # )
