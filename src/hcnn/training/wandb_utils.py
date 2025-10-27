import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb


def create_and_log_visualizations(train, epoch, samples_per_axis=300):
    """
    Create and log visualizations of the model to wandb.

    Args:
        train: The training bundle containing the models.
        epoch: The current epoch number.
        samples_per_axis: Number of samples per axis for the visualization grid.
    """
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
        states = jnp.stack((x_coord.flatten(), y_coord.flatten()), axis=-1)
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
            f"Skipping visualization for {train.policy.state_dim}D state space"
        )
        return

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
    if train.policy.action_dim == 1:
        train.policy.eval()
        actions = (
            train.policy(states)
            .squeeze()
            .reshape(samples_per_axis, samples_per_axis)
        )
        train.policy.train()

        plt.figure(figsize=(9, 7))
        plt.pcolormesh(x, y, actions, cmap="viridis", shading="gouraud")
        plt.colorbar()
        wandb.log(step=epoch, data={"policy-actions": wandb.Image(plt)})
        plt.close()
