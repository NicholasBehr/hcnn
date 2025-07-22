import os
from functools import partial

import matplotlib
import mujoco

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
from hcnn.system_costs import SystemCostsMuJoCo, SystemCostsQuadratic
from hcnn.system_dynamics import SystemDynamicsLinear, SystemDynamicsMuJoCo
from hcnn.training.dropout_mlp import DropoutMLP
from hcnn.training.loss_and_grad import LossAndGrad, ModelStates

opt_map = {
    "adabelief": optax.adabelief,
    "adagrad": optax.adagrad,
    "adam": optax.adam,
    "adamw": optax.adamw,
    "nadamw": optax.nadamw,
}


def build_sys_components(config):
    if config.system_name == "double_integrator":
        sys_dynamics = SystemDynamicsLinear.factory(
            state_matrix=jnp.array([[1, 1], [0, 1]]),
            action_matrix=jnp.array([[0.5], [1]]),
        )
        sys_costs = SystemCostsQuadratic.factory(
            state_cost_matrix=jnp.array([[5, 0], [0, 5]]),
            action_cost_matrix=jnp.array([[1]]),
        )
        sys_constraints = SystemConstraintsBox.factory(
            state_l=jnp.array([-10, -10]),
            state_u=jnp.array([+10, +10]),
            action_l=jnp.array([-1]),
            action_u=jnp.array([+1]),
        )
        return sys_dynamics, sys_costs, sys_constraints
    elif config.system_name == "chaotic_pendulum":
        chaotic_pendulum = r"""
        <mujoco>
        <option
            timestep=".02"
            integrator="implicitfast" solver="CG" tolerance="1e-4">
            <flag contact="disable" constraint="disable" sensor="disable"/>
        </option>

        <default>
            <joint type="hinge" axis="0 -1 0" limited="false" range="-150 150"/>
            <geom type="capsule" size=".02"/>
        </default>

        <worldbody>
            <light pos="0 -.4 1"/>
            <camera name="fixed" pos="0 -1.5 0" xyaxes="1 0 0 0 0 1"/>

            <body name="red" pos="0 0 .2">
                <joint name="root" />
                <geom fromto="0 0 0 0 0 -.25" rgba="1 0 0 1"/>
                <body name="green" pos="0 0 -.25">
                    <joint/>
                    <geom fromto="0 0 0 0 0 -.25" rgba="0 1 0 1"/>
                    <body name="blue" pos="0 0 -.25">
                        <joint/>
                        <geom fromto="0 0 0 0 0 -.25" rgba="0 0 1 1"/>
                    </body>
                </body>
            </body>
        </worldbody>

        <actuator>
            <motor name="joint_motor" joint="root"/>
        </actuator>
        </mujoco>
        """

        model = mujoco.MjModel.from_xml_string(chaotic_pendulum)  # type: ignore
        mjx_model = mujoco.mjx.put_model(model)  # type: ignore

        sys_dynamics = SystemDynamicsMuJoCo.factory(
            mjx_model=mjx_model,
            control_timestep=0.1,
        )
        q_identity = jnp.array([1.0, 0.0, 0.0, 0.0])
        q_swung_up = jnp.array([0.0, 0.0, 1.0, 0.0])
        quaternion_ref = jnp.vstack(
            [
                q_identity,  # world body
                q_swung_up,  # red link
                q_swung_up,  # green link
                q_swung_up,  # blue link
            ]
        )

        sys_costs = SystemCostsMuJoCo.factory(
            model=mjx_model,
            cost_position=jnp.array([0.0, 0.0, 0.0, 0.0]),
            cost_orientation=jnp.array([0.0, 1.0, 1.0, 1.0]),
            position_reference=jnp.zeros((model.nbody, 3)),
            quaternion_reference=quaternion_ref,
        )
        sys_constraints = SystemConstraintsBox.factory(
            state_l=-jnp.inf * jnp.ones(sys_dynamics.state_dim),
            state_u=jnp.inf * jnp.ones(sys_dynamics.state_dim),
            action_l=-2 * jnp.ones(sys_dynamics.action_dim),
            action_u=2 * jnp.ones(sys_dynamics.action_dim),
        )
        return sys_dynamics, sys_costs, sys_constraints
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
            activation=config.policy_activation,
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
        activation=config.cost_to_go_activation,
        rngs=rngs,
    )


def build_relax_to_go(config, state_dim, rngs) -> DropoutMLP:
    return DropoutMLP(
        layer_sizes=[state_dim]
        + [config.relax_to_go_hidden_dim] * config.relax_to_go_hidden_depth
        + [1],
        dropout_rate=config.relax_to_go_dropout_rate,
        activation=config.relax_to_go_activation,
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


def build_optimizer(model, name, learning_rate):
    if name in opt_map:
        return nnx.Optimizer(model, opt_map[name](learning_rate=learning_rate))
    else:
        raise ValueError(f"Unknown optimizer: {name}")


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
            minval=-10,
            maxval=+10,
        )
        jax_keys = jax.random.split(split_key, batchsize)

        # Pre-allocate model states to reduce overhead
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

        # Update metrics with batched operations
        train.metrics.update(
            policy_loss=loss.policy,
            cost_to_go_loss=loss.cost_to_go,
            relax_to_go_loss=loss.relax_to_go,
            aggregate_loss=aggregate_loss,
            primal_residual=info.primal_residual,
            dual_residual=info.dual_residual,
            solver_iterations=info.solver_iterations,
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

        # Build system model components
        sys_dynamics, sys_costs, sys_constraints = build_sys_components(config)

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
            # model evaluation metrics
            policy_loss=nnx.metrics.Average("policy_loss"),
            cost_to_go_loss=nnx.metrics.Average("cost_to_go_loss"),
            relax_to_go_loss=nnx.metrics.Average("relax_to_go_loss"),
            aggregate_loss=nnx.metrics.Average("aggregate_loss"),
            # solver metrics
            primal_residual=nnx.metrics.Average("primal_residual"),
            dual_residual=nnx.metrics.Average("dual_residual"),
            solver_iterations=nnx.metrics.Average("solver_iterations"),
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

        # Pre-warm the JIT compilation with a dummy forward pass
        print("Pre-warming JIT compilation...")
        _ = train_epoch(
            config=config,
            train_graphdef=train_graphdef,
            train_state=train_state,
        )
        print("JIT compilation complete, starting training...")

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
                axis_samples = jnp.linspace(-14, 14, samples_per_axis)
                x_coord, y_coord = jnp.meshgrid(
                    axis_samples,
                    axis_samples,
                    indexing="xy",
                )

                # Flatten the grid to create a list of states
                # This is necessary for batch processing in JAX
                # We stack the coordinates to create a 2D array of states
                # Each row corresponds to a state (x, y)
                states = jnp.stack(
                    (x_coord.flatten(), y_coord.flatten()), axis=-1
                )
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

    jax.config.update("jax_platform_name", "cpu")

    # Specify the project name and configuration
    # This can be overridden by wandb.agent when running a sweep
    project = "chaotic_pendulum"
    config = {
        "system_name": "chaotic_pendulum",
        "seed": 0,
        "policy_name": "dropout_mlp",
        "policy_hidden_dim": 3,
        "policy_hidden_depth": 1,
        "policy_dropout_rate": 0.0,
        "policy_activation": "leaky_relu",
        "policy_intervention_cost": 0.5,
        "cost_to_go_hidden_dim": 3,
        "cost_to_go_hidden_depth": 1,
        "cost_to_go_dropout_rate": 0.0,
        "cost_to_go_activation": "leaky_relu",
        "relax_to_go_hidden_dim": 3,
        "relax_to_go_hidden_depth": 1,
        "relax_to_go_dropout_rate": 0.0,
        "relax_to_go_activation": "leaky_relu",
        "projection_name": "no_substitution_no_terminal",
        "projection_horizon": 2,
        "projection_relaxation_penalty": 50,
        "solver_name": "osqp",
        "solver_alpha": 1,
        "solver_sigma": 0.1,
        "solver_rho_equality": 0.4,
        "solver_rho_inequality": 1.5,
        "solver_max_iter": 10,
        "solver_tol": 1e-3,
        "optimizer_policy_name": "adamw",
        "optimizer_policy_learning_rate": 1e-1,
        "optimizer_cost_name": "adamw",
        "optimizer_cost_learning_rate": 1e-1,
        "optimizer_relax_name": "adamw",
        "optimizer_relax_learning_rate": 1e-1,
        "batchsize": 1,
        "batches_per_epoch": 1,
        "td_gamma": 0.95,
        "gae_lambda": 0.99,
        "epochs": 5,
    }

    # Add this before your sweep_configuration definition:
    optimizer_keys = list(opt_map.keys())

    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "solver_iterations"},
        # "early_terminate": {"type": "hyperband", "min_iter": 20, "eta": 1.5},
        "parameters": {
            "system_name": {"value": "double_integrator"},
            "seed": {"value": 42},
            "policy_name": {"value": "dropout_mlp"},
            "policy_hidden_dim": {"value": 3},
            "policy_hidden_depth": {"value": 3},
            "policy_dropout_rate": {"value": 0.1},
            "policy_activation": {"value": "leaky_relu"},
            "policy_intervention_cost": {"value": 0.5},
            # Cost-to-go
            "cost_to_go_hidden_dim": {"value": 3},
            "cost_to_go_hidden_depth": {"value": 3},
            "cost_to_go_dropout_rate": {"value": 0.0},
            "cost_to_go_activation": {"value": "leaky_relu"},
            # Relax-to-go
            "relax_to_go_hidden_dim": {"value": 4},
            "relax_to_go_hidden_depth": {"value": 3},
            "relax_to_go_dropout_rate": {"value": 0.0},
            "relax_to_go_activation": {"value": "leaky_relu"},
            "projection_name": {"value": "no_substitution_no_terminal"},
            "projection_horizon": {"value": 20},
            "projection_relaxation_penalty": {"value": 50},
            "solver_name": {"value": "shallow"},
            "solver_alpha": {"min": 0.1, "max": 1.9},
            "solver_sigma": {
                "distribution": "log_uniform_values",
                "max": 5,
                "min": 0.01,
            },
            "solver_rho_equality": {
                "distribution": "log_uniform_values",
                "max": 2,
                "min": 0.01,
            },
            "solver_rho_inequality": {
                "distribution": "log_uniform_values",
                "max": 2,
                "min": 0.01,
            },
            "solver_max_iter": {"value": 5e5},
            "solver_tol": {"value": 1e-4},
            "optimizer_policy_name": {"value": "adamw"},
            "optimizer_policy_learning_rate": {"value": 1e-2},
            "optimizer_cost_name": {"value": "adamw"},
            "optimizer_cost_learning_rate": {"value": 1e-2},
            "optimizer_relax_name": {"value": "adamw"},
            "optimizer_relax_learning_rate": {"value": 1e-2},
            "batchsize": {"value": 32},
            "batches_per_epoch": {"value": 1},
            "td_gamma": {"value": 0.95},
            "gae_lambda": {"value": 0.99},
            "epochs": {"value": 3},
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
