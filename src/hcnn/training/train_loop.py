import os
import shutil
import tempfile
from functools import partial

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

import wandb
from hcnn.optimization import OptimizationInfo
from hcnn.projection import ProjectionLayer
from hcnn.state_to_rollout.state_to_rollout_base import StateToRolloutBase
from hcnn.system_costs.system_costs_base import SystemCostsBase
from hcnn.training.train_bundle import TrainBundle

# from hcnn.training.visualization import visualize_receding_horizon


def loss_fn(
    state_to_rollout: StateToRolloutBase,
    initial_states: jax.Array,
    projection_layer: ProjectionLayer,
    sys_costs: SystemCostsBase,
    projection_cost=1.0,
):
    """Compute the loss for the training step."""
    rollout = state_to_rollout(initial_states)
    rollout_proj, _, info = projection_layer(rollout, initial_states)
    rollout_proj_cost = sys_costs(rollout_proj.states, rollout_proj.actions)

    # Compute the cost and difference
    cost = jnp.mean(rollout_proj_cost)
    state_diff = jnp.mean(
        jnp.abs(rollout.states - jax.lax.stop_gradient(rollout_proj.states))
    )
    action_diff = jnp.mean(
        jnp.abs(rollout.actions - jax.lax.stop_gradient(rollout_proj.actions))
    )
    diff = state_diff + action_diff

    # Combine cost and difference into a single loss
    loss = cost + projection_cost * diff

    # Aggregate the optimization info
    info = OptimizationInfo(
        primal_residual=jnp.mean(info.primal_residual),
        dual_residual=jnp.mean(info.dual_residual),
        iterations_lin=jnp.mean(info.iterations_lin),
        iterations_nonlin=jnp.mean(info.iterations_nonlin),
        converged_flag=jnp.mean(info.converged_flag),
    )

    return loss, (loss, cost, diff, info)


@partial(jax.jit, static_argnames=["batches_per_epoch", "train_graphdef"])
def train_epoch(
    train_graphdef: nnx.GraphDef,
    train_state: nnx.State,
    batches_per_epoch: int = 1,
    projection_cost: float = 1.0,
) -> tuple[nnx.State, dict["str", jax.Array]]:
    """
    Train a single epoch of the model.
    This function performs a series of training steps, updating the model
    parameters based on the computed gradients and losses.
    It uses JAX's just-in-time compilation to optimize performance.

    Args:
        config: Configuration object containing training parameters.
        train_graphdef: The graph definition of the training bundle.
        train_state: The current state of the training bundle.

    Returns:
        The updated train_state after completing the epoch.
    """
    # Reset metrics at the start of each epoch
    train = nnx.merge(train_graphdef, train_state)
    train.metrics.reset()
    train_state = nnx.state(train)

    def train_step(_, train_state: nnx.State) -> nnx.State:
        train = nnx.merge(train_graphdef, train_state)

        grads, aux = nnx.grad(loss_fn, has_aux=True)(
            train.state_to_rollout,
            train.sampler(),
            train.projection_layer,
            train.system.sys_costs,
            projection_cost,
        )

        # Apply the gradients to the optimizer
        train.optimizer.update(grads)

        # Unpack the auxiliary values
        loss, cost, diff, info = aux

        # Update metrics
        train.metrics.update(
            cost=cost,
            difference=diff,
            loss=loss,
            primal_residual=info.primal_residual,
            dual_residual=info.dual_residual,
            solver_iter_lin=info.iterations_lin,
            solver_iter_nonlin=info.iterations_nonlin,
            solver_converged=info.converged_flag,
        )

        return nnx.state(train)

    # Iterate over training steps
    train_state = jax.lax.fori_loop(
        lower=0,
        upper=batches_per_epoch,
        body_fun=train_step,
        init_val=train_state,
    )

    # Compute metrics
    train = nnx.merge(train_graphdef, train_state)
    metrics = train.metrics.compute()
    train_state = nnx.state(train)

    return train_state, metrics


def save_train_state(train_state: nnx.State):
    """
    Save the current training state to Weights & Biases using Orbax.
    This function serializes the train_state and uploads it to W&B as an
    artifact. Each call creates a new version of the same artifact.

    Args:
        train_state: The current state of the training bundle.
    """
    # Create a persistent temporary directory for the checkpoint
    temp_dir = tempfile.mkdtemp(prefix="hcnn_checkpoint_")

    try:
        checkpoint_dir = os.path.join(temp_dir, "train_state")

        # Save using Orbax StandardCheckpointer
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(checkpoint_dir, train_state)

        # Wait for save to complete
        checkpointer.wait_until_finished()

        # Verify the checkpoint directory exists and has content
        if not os.path.exists(checkpoint_dir):
            raise RuntimeError(
                f"Checkpoint directory not created: {checkpoint_dir}"
            )

        if not os.listdir(checkpoint_dir):
            raise RuntimeError(
                f"Checkpoint directory is empty: {checkpoint_dir}"
            )

        # Create a wandb artifact with a run-specific name
        # This ensures each run has its own artifacts
        if wandb.run is None:
            raise RuntimeError("No active wandb run found")
        run_id = wandb.run.id
        artifact = wandb.Artifact(
            name=f"model_checkpoint_{run_id}",
            type="model",
            description="HCNN training state checkpoint",
            metadata={
                "model_type": "hcnn",
                "run_id": run_id,
            },
        )

        # Add the entire checkpoint directory to the artifact
        artifact.add_dir(checkpoint_dir)

        # Log the artifact to wandb
        wandb.log_artifact(artifact)

    finally:
        # Clean up the temporary directory after wandb has processed it
        try:
            shutil.rmtree(temp_dir)
        except OSError:
            # If cleanup fails, just warn but don't fail the training
            print(
                f"Warning: Could not clean up temporary directory: {temp_dir}"
            )


def train(project=None, config=None):
    # When called by wandb.agent, project is handled by the agent
    # and config comes from the sweep
    with wandb.init(project=project, config=config):
        # If called by wandb.agent,
        # this config will be set by Sweep Controller
        config = wandb.config

        # Build TrainBundle
        train = TrainBundle.factory_from_config(config)
        train_graphdef, train_state = nnx.split(train)

        for epoch in range(config.epochs):
            # Advance the training by one epoch
            train_state, metrics = train_epoch(
                train_graphdef=train_graphdef,
                train_state=train_state,
                batches_per_epoch=config.batches_per_epoch,
            )

            # Print metrics in a nicely formatted table
            if epoch == 0:
                header = f"{'Epoch':<8}" + "".join(
                    [f"{k:<20}" for k in metrics.keys()]
                )
                print(header)
            values = f"{epoch:<8}" + "".join(
                [f"{float(v):<20.3f}" for v in metrics.values()]
            )
            print(values)

            # Log to WandB
            wandb.log(step=epoch, data=metrics)
            # if epoch % 25 == 0 or epoch == config.epochs - 1:
            #     visualize_receding_horizon(
            #         train_bundle=nnx.merge(train_graphdef, train_state),
            #         steps=50,
            #         initial_state=3 * jnp.ones(train.policy.state_dim),
            #     )
            #     save_train_state(train_state)


if __name__ == "__main__":
    print("This is a training loop with Weights & Biases integration.")

    # restrict access to one GPU (check availability nvidia-smi)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    jax.config.update("jax_debug_nans", False)
    # jax.config.update("jax_platform_name", "cpu")  # Uncomment to run on CPU

    # Specify the project name and configuration
    # This can be overridden by wandb.agent when running a sweep
    project = "double_integrator_mpc"

    config = {
        "system_name": "double_integrator",
        "seed": 0,
        # Neural Network
        "nn_name": "action",
        "nn_hidden_dim": 16,
        "nn_hidden_depth": 8,
        "nn_dropout_rate": 0.1,
        "nn_activation": "leaky_relu",
        # Optimizer
        "optimizer_learning_rate": 1e-2,
        "optimizer_b1": 0.9,
        "optimizer_b2": 0.999,
        "optimizer_weight_decay": 1e-4,
        # Solver
        "solver_name": "shallow",
        "solver_alpha": 1.6,
        "solver_sigma": 1e-3,
        "solver_rho_equality": 1,
        "solver_rho_inequality": 1,
        "solver_max_iter_nonlin": 1,
        "solver_max_iter_lin": 5e3,
        "solver_tol": 1e-1,
        # Projection
        "projection_horizon": 30,
        # Training parameters
        "batch_size": 512,
        "batches_per_epoch": 16,
        "epochs": 50,
    }
    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "residual"},
        # "early_terminate": {"type": "hyperband", "min_iter": 20, "eta": 1.5},
        "parameters": {
            "system_name": {"value": project},
            "seed": {"value": 0},
            # Policy
            "policy_name": {"value": "dropout_mlp"},
            "policy_hidden_dim": {"value": 5},
            "policy_hidden_depth": {"value": 5},
            "policy_dropout_rate": {"value": 0.15},
            "policy_activation": {"value": "leaky_relu"},
            "policy_intervention_cost": {"value": 0.5},
            # Cost-to-go and Relax-to-go
            "x_to_go_hidden_dim": {"value": 5},
            "x_to_go_hidden_depth": {"value": 5},
            "x_to_go_dropout_rate": {"value": 0.15},
            "x_to_go_activation": {"value": "leaky_relu"},
            # Projection
            "projection_name": {"value": "no_substitution_no_terminal"},
            "projection_horizon": {"value": 20},
            "projection_relaxation_penalty": {"value": 1e4},
            # Solver
            "solver_name": {"value": "osqp"},
            "solver_alpha": {
                "distribution": "uniform",
                "max": 1.8,
                "min": 0.2,
            },
            "solver_sigma": {
                "distribution": "log_uniform_values",
                "max": 1e3,
                "min": 1e-3,
            },
            "solver_rho_equality": {
                "distribution": "log_uniform_values",
                "max": 1e3,
                "min": 1e-3,
            },
            "solver_rho_inequality": {
                "distribution": "log_uniform_values",
                "max": 1e3,
                "min": 1e-3,
            },
            "solver_max_iter_nonlin": {"value": 1},
            "solver_max_iter_lin": {"value": 1e3},
            "solver_tol": {"value": 1e-3},
            # Optimizers
            "optimizer_policy_name": {"value": "adamw"},
            "optimizer_policy_learning_rate": {"value": 1e-3},
            "optimizer_cost_name": {"value": "adamw"},
            "optimizer_cost_learning_rate": {"value": 1e-1},
            "optimizer_relax_name": {"value": "adamw"},
            "optimizer_relax_learning_rate": {"value": 1e-1},
            # Training parameters
            "batchsize": {"value": 64},
            "batches_per_epoch": {"value": 1},
            "td_gamma": {"value": 0.95},
            "gae_lambda": {"value": 0.95},
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
    #     count=20,
    # )
