import os
import shutil
import tempfile
from functools import partial

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

import wandb
from hcnn.training.loss_and_grad import ModelStates
from hcnn.training.train_bundle import TrainBundle
from wandb.sdk.wandb_config import Config as WandbConfig


@partial(
    jax.jit,
    static_argnames=["config", "train_graphdef"],
    donate_argnums=(2,),  # Donate train_state to reduce memory overhead
)
def train_epoch(
    config: WandbConfig,
    train_graphdef: nnx.GraphDef,
    train_state: nnx.State,
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

    # Pre-compile scalar constants to avoid recompilation
    batchsize = config.batchsize
    td_gamma = config.td_gamma
    gae_lambda = config.gae_lambda
    intervention_cost = config.policy_intervention_cost

    # Reset metrics at the start of each epoch
    train = nnx.merge(train_graphdef, train_state)
    train.metrics.reset()
    train_state = nnx.state(train)

    def train_step(step_idx: int, train_state: nnx.State) -> nnx.State:
        train = nnx.merge(train_graphdef, train_state)

        # Use step-dependent but deterministic sampling to reduce
        # compilation variance
        step_key = jax.random.fold_in(train.rngs.default(), step_idx)
        sample_key, split_key = jax.random.split(step_key)

        model_states = ModelStates(
            policy=nnx.state(train.policy),
            relax_to_go=nnx.state(train.relax_to_go),
            cost_to_go=nnx.state(train.cost_to_go),
        )
        # TODO, add a more sophisticated sampler
        initial_states = jax.random.uniform(
            sample_key,
            (batchsize, train.policy.state_dim),
            minval=-2 * jnp.pi,
            maxval=+2 * jnp.pi,
        )
        jax_keys = jax.random.split(split_key, batchsize)

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

        # Compute maximum absolute gradient value for monitoring and clipping
        grad_leaves = jax.tree_util.tree_leaves(grad.policy)
        max_policy_gradient = jnp.max(
            jnp.array([jnp.max(jnp.abs(leaf)) for leaf in grad_leaves])
        )
        mean_policy_gradient = jnp.mean(
            jnp.array([jnp.mean(jnp.abs(leaf)) for leaf in grad_leaves])
        )

        # Update metrics with batched operations
        train.metrics.update(
            policy_loss=loss.policy,
            cost_to_go_loss=loss.cost_to_go,
            relax_to_go_loss=loss.relax_to_go,
            primal_residual=info.primal_residual,
            dual_residual=info.dual_residual,
            solver_iterations=info.solver_iterations,
            max_policy_gradient=max_policy_gradient,
            mean_policy_gradient=mean_policy_gradient,
        )

        # Update optimizers with pre-allocated gradients
        train.opt_policy.update(grad.policy)
        train.opt_relax.update(grad.relax_to_go)
        train.opt_cost.update(grad.cost_to_go)

        return nnx.state(train)

    # Iterate over training steps
    train_state = jax.lax.fori_loop(
        lower=0,
        upper=config.batches_per_epoch,
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
        train_graphdef: The graph definition of the training bundle.
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


def train(project, config):
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
                config=config,
                train_graphdef=train_graphdef,
                train_state=train_state,
            )

            # Print metrics in a nicely formatted table
            if epoch == 0:
                header = f"{'Epoch':<8}" + "".join(
                    [f"{k:<20}" for k in metrics.keys()]
                )
                print(header)
            values = f"{epoch:<8}" + "".join(
                [f"{float(v):<20.2f}" for v in metrics.values()]
            )
            print(values)

            # Log to WandB
            wandb.log(step=epoch, data=metrics)

            if epoch % 25 == 0 or epoch == config.epochs - 1:
                save_train_state(train_state)


if __name__ == "__main__":
    print("This is a training loop with Weights & Biases integration.")

    # restrict access to one GPU (check availability nvidia-smi)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_platform_name", "cpu")

    # Specify the project name and configuration
    # This can be overridden by wandb.agent when running a sweep
    project = "double_pendulum_2"
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
        "optimizer_policy_learning_rate": 1e-2,
        "optimizer_cost_name": "adamw",
        "optimizer_cost_learning_rate": 1e-1,
        "optimizer_relax_name": "adamw",
        "optimizer_relax_learning_rate": 1e-1,
        "batchsize": 5,
        "batches_per_epoch": 1,
        "td_gamma": 0.9,
        "gae_lambda": 0.8,
        "epochs": 10,
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
