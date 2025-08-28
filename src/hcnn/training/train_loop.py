import os
import shutil
import tempfile

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

import wandb
from hcnn.optimization_layer import MPCLayer, ProjectionLayer
from hcnn.state_to_rollout import StateToRolloutBase
from hcnn.system_constraints import SystemConstraintsBox
from hcnn.system_costs import SystemCostsBase, SystemCostsQuadratic
from hcnn.system_dynamics import SystemDynamicsLinear
from hcnn.training.train_bundle import TrainBundle
from hcnn.training.visualization import (
    visualize_receding_horizon,
    visualize_receding_horizon_mpc,
)


def loss_fn(
    state_to_rollout: StateToRolloutBase,
    initial_states: jax.Array,
    projection_layer: ProjectionLayer,
    sys_costs: SystemCostsBase,
    alignment_weight: float = 1.0,
):
    """Compute the loss for the training step."""
    rollout = state_to_rollout(initial_states)
    rollout_proj, info = projection_layer(rollout, initial_states)

    # Compute the cost
    cost = jnp.mean(sys_costs(rollout_proj.states, rollout_proj.actions))

    # Compute squared distances per sample
    state_sqr_dst = (
        rollout.states - jax.lax.stop_gradient(rollout_proj.states)
    ) ** 2
    action_sqr_dst = (
        rollout.actions - jax.lax.stop_gradient(rollout_proj.actions)
    ) ** 2

    # Take mean of the conditionally-selected squared distances
    sqr_dst = jnp.mean(state_sqr_dst) + jnp.mean(action_sqr_dst)

    # Apply alignment weight
    alignment_term = alignment_weight * sqr_dst

    # Combine cost and alignment term into a single loss
    loss = cost + alignment_term

    # Aggregate the optimization info
    info = jax.tree.map(lambda x: jnp.mean(x.astype(float)), info)

    return loss, (loss, cost, sqr_dst, info)


@jax.jit
def train_epoch(
    train_graphdef: nnx.GraphDef,
    train_state: nnx.State,
    batches_per_epoch: int = 1,
    alignment_weight: float = 1e-3,
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
            state_to_rollout=train.state_to_rollout,
            initial_states=train.sampler(),
            projection_layer=train.projection_layer,
            sys_costs=train.system.sys_costs,
            alignment_weight=alignment_weight,
        )

        # Apply the gradients to the optimizer
        train.optimizer.update(train.state_to_rollout, grads)

        # Unpack the auxiliary values
        loss, cost, sqr_dst, info = aux

        # Update metrics
        train.metrics.update(
            loss=loss,
            cost=cost,
            sqr_dst=sqr_dst,
            gm_residual=jnp.sqrt(info.primal_residual * info.dual_residual),
            **vars(info),
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
                alignment_weight=config.alignment_weight,
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
            if (epoch % 10 == 0 and epoch > 0) or epoch == config.epochs - 1:
                state_dim = train.system.sys_dynamics.state_dim
                train_bundle = nnx.merge(train_graphdef, train_state)

                # Always visualize the projection layer
                visualize_receding_horizon(
                    train_bundle=train_bundle,
                    steps=50,
                    initial_state=3 * jnp.ones(state_dim),
                    wandb_step=epoch,
                )

            if epoch == config.epochs - 1:
                save_train_state(train_state)

        # Add MPC visualization if system dynamics are linear
        if (
            isinstance(train.system.sys_dynamics, SystemDynamicsLinear)
            and isinstance(train.system.sys_costs, SystemCostsQuadratic)
            and isinstance(
                train.system.sys_constraints,
                SystemConstraintsBox,
            )
        ):
            # Create MPC layer for comparison
            mpc_layer = MPCLayer.factory(
                sys_dynamics=train.system.sys_dynamics,
                sys_constraints=train.system.sys_constraints,
                sys_costs=train.system.sys_costs,
                horizon=train.projection_layer.projection_problem.horizon,
                sqp_parameters=train.projection_layer.sqp_parameters,
            )

            # Visualize MPC for comparison
            visualize_receding_horizon_mpc(
                mpc_layer=mpc_layer,
                sys_dynamics=train.system.sys_dynamics,
                steps=50,
                initial_state=3 * jnp.ones(train.system.sys_dynamics.state_dim),
                wandb_step=config.epochs - 1,
            )


if __name__ == "__main__":
    print("This is a training loop with Weights & Biases integration.")

    # restrict access to one GPU (check availability nvidia-smi)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    jax.config.update("jax_debug_nans", False)
    # jax.config.update("jax_platform_name", "cpu")  # Uncomment to run on CPU

    # Specify the project name and configuration
    # This can be overridden by wandb.agent when running a sweep
    project = "bike"

    config = {
        "system_name": project,
        "seed": 0,
        # Neural Network
        "nn_name": "actions",
        "nn_hidden_dim": 64,
        "nn_hidden_depth": 3,
        "nn_dropout_rate": 0.2,
        "nn_activation": "leaky_relu",
        # Optimizer
        "optimizer_learning_rate": 1e-2,
        # Solver
        "solver_alpha": 1.6,
        "solver_sigma": 1e-6,
        "solver_rho_base": 10,
        "rho_equality_mult": 1e3,
        "solver_tol_prim": 1e-4,
        "solver_tol_dual": 1e-2,
        "solver_epsilon": 1e-4,
        "solver_max_iter_lin": 1e3,
        "solver_max_iter_nonlin": 5,
        # Projection
        "projection_horizon": 10,
        # Training parameters
        "batch_size": 64,
        "batches_per_epoch": 1,
        "epochs": 300,
        "alignment_weight": 1e-3,
    }
    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "cost"},
        "early_terminate": {"type": "hyperband", "min_iter": 20, "eta": 1.5},
        "parameters": {
            "system_name": {"value": "double_integrator"},
            "seed": {"value": 0},
            # Neural Network
            "nn_name": {"value": "scan"},
            "nn_hidden_dim": {"value": 5},
            "nn_hidden_depth": {"value": 5},
            "nn_dropout_rate": {"value": 0.2},
            "nn_activation": {"value": "leaky_relu"},
            # Optimizer
            "optimizer_learning_rate": {
                "distribution": "log_uniform_values",
                "max": 1e-1,
                "min": 1e-4,
            },
            "optimizer_b1": {
                "distribution": "uniform",
                "max": 0.99,
                "min": 0.8,
            },
            "optimizer_b2": {
                "distribution": "uniform",
                "max": 0.999,
                "min": 0.9,
            },
            "optimizer_weight_decay": {
                "distribution": "log_uniform_values",
                "max": 1e-1,
                "min": 1e-4,
            },
            # Solver
            "solver_name": {"value": "osqp"},
            "solver_alpha": {"value": 1.6},
            "solver_sigma": {"value": 15},
            "solver_rho_base": {"value": 100},
            "solver_rho_eq_mult": {"value": 0.05},
            "solver_max_iter_nonlin": {"value": 1},
            "solver_max_iter_lin": {"value": 5e3},
            "solver_tol": {"value": 0.05},
            # Projection
            "projection_horizon": {"value": 20},
            "projection_relaxation_penalty": {"value": 1e4},
            # Training parameters
            "batch_size": {"value": 512},
            "batches_per_epoch": {"value": 16},
            "epochs": {"value": 100},
            "alignment_weight": {
                "distribution": "log_uniform_values",
                "max": 1e-1,
                "min": 1e-4,
            },
        },
    }

    train(project=project, config=config)

    # sweep_id = "ak65u4jx"  # Replace with your sweep ID if you have one
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
