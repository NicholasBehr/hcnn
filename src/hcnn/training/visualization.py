import jax
import jax.numpy as jnp
import plotly.graph_objects as go
import wandb
from flax import nnx
from plotly.subplots import make_subplots

from .train_bundle import TrainBundle


@nnx.jit
def sample_rollout(train_bundle: TrainBundle, initial_states: jax.Array):
    rollout = train_bundle.state_to_rollout(initial_states)
    rollout_proj, _ = train_bundle.projection_layer(rollout, initial_states)
    return rollout_proj


def _create_visualization_subplot(
    state_dim: int, action_dim: int
) -> tuple[go.Figure, int]:
    """Create the subplot structure for receding horizon visualization."""
    total_plots = state_dim + action_dim
    fig = make_subplots(rows=total_plots, cols=1, vertical_spacing=0.1)
    return fig, total_plots


def _add_planned_trajectory(
    fig: go.Figure,
    rollout_time_steps: jax.Array,
    action_time_steps: jax.Array,
    planned_states: jax.Array,
    planned_actions: jax.Array,
    state_dim: int,
    action_dim: int,
    total_plots: int,
    step: int,
    prefix: str = "",
):
    """Add planned trajectory traces to the figure."""
    # Colors for visualization
    light_color = "rgba(100, 149, 237, 0.3)"  # Light blue for planned
    action_light_color = (
        "rgba(255, 165, 0, 0.3)"  # Light orange for planned actions
    )

    # Add planned state trajectory (light color)
    for state_idx in range(state_dim):
        fig.add_trace(
            go.Scatter(
                x=rollout_time_steps,
                y=planned_states[:, state_idx],
                mode="lines",
                line=dict(color=light_color, width=1),
                name=(
                    f"{prefix}Planned States"
                    if step == 0 and state_idx == 0
                    else None
                ),
                showlegend=True if step == 0 and state_idx == 0 else False,
            ),
            row=state_idx + 1,
            col=1,
        )

    # Add planned action trajectory (light orange)
    for action_idx in range(action_dim):
        fig.add_trace(
            go.Scatter(
                x=action_time_steps,
                y=planned_actions[
                    :-1, action_idx
                ],  # Exclude last action as it's not used
                mode="lines",
                line=dict(color=action_light_color, width=1),
                name=f"{prefix}Planned Actions"
                if step == 0 and action_idx == 0
                else None,
                showlegend=True if step == 0 and action_idx == 0 else False,
            ),
            row=state_dim + action_idx + 1,
            col=1,
        )


def _add_actual_trajectory(
    fig: go.Figure,
    step: int,
    current_state: jax.Array,
    next_state: jax.Array,
    first_action: jax.Array,
    state_dim: int,
    action_dim: int,
    total_plots: int,
    prefix: str = "",
):
    """Add actual trajectory traces to the figure."""
    # Colors for visualization
    dark_color = "rgba(0, 0, 139, 1.0)"  # Dark blue for actual
    action_dark_color = (
        "rgba(255, 140, 0, 1.0)"  # Dark orange for actual actions
    )

    # Add the actual state transition (dark color)
    for state_idx in range(state_dim):
        fig.add_trace(
            go.Scatter(
                x=[step, step + 1],
                y=[current_state[state_idx], next_state[state_idx]],
                mode="lines+markers",
                line=dict(color=dark_color, width=3),
                marker=dict(size=6, color=dark_color),
                name=f"{prefix}Actual States"
                if step == 0 and state_idx == 0
                else None,
                showlegend=True if step == 0 and state_idx == 0 else False,
            ),
            row=state_idx + 1,
            col=1,
        )

    # Add the actual action taken (dark orange)
    for action_idx in range(action_dim):
        fig.add_trace(
            go.Scatter(
                x=[step],
                y=[first_action[action_idx]],
                mode="markers",
                marker=dict(size=6, color=action_dark_color),
                name=f"{prefix}Actual Actions"
                if step == 0 and action_idx == 0
                else None,
                showlegend=True if step == 0 and action_idx == 0 else False,
            ),
            row=state_dim + action_idx + 1,
            col=1,
        )


def _finalize_figure(
    fig: go.Figure,
    state_dim: int,
    action_dim: int,
    total_plots: int,
    steps: int,
    title: str,
):
    """Finalize the figure layout and axes."""
    # Update layout with padding
    fig.update_layout(
        height=500 * total_plots,
        margin=dict(l=60, r=60, t=120, b=60),
        title=title,
    )

    # Update x-axis labels for all subplots with padding
    for i in range(state_dim):
        fig.update_xaxes(title_text="Time Step", row=i + 1, col=1)
        fig.update_yaxes(
            title_text=f"State {i + 1} Value", row=i + 1, col=1, automargin=True
        )

    for i in range(action_dim):
        fig.update_xaxes(title_text="Time Step", row=state_dim + i + 1, col=1)
        fig.update_yaxes(
            title_text=f"Action {i + 1} Value",
            row=state_dim + i + 1,
            col=1,
            automargin=True,
        )


def visualize_receding_horizon(
    train_bundle: TrainBundle,
    steps: int = 50,
    initial_state: jax.Array | None = None,
    wandb_step: int | None = None,
):
    """
    Visualize the receding horizon evolution using ProjectionLayer.
    At each step, show the full planned rollout and the actual step taken.
    """
    # make everything in bundle deterministic
    train_bundle.eval()

    sys_dynamics = train_bundle.system.sys_dynamics
    state_dim = sys_dynamics.state_dim
    action_dim = sys_dynamics.action_dim
    horizon = train_bundle.projection_layer.projection_problem.horizon

    # If no initial state is provided, use zero state
    if initial_state is None:
        initial_state = jnp.zeros(sys_dynamics.state_dim)

    # Create subplots
    fig, total_plots = _create_visualization_subplot(state_dim, action_dim)

    # Track the actual trajectory taken
    current_state = initial_state

    # Simulate receding horizon for the specified number of steps
    for step in range(steps):
        # Get rollout from current state
        rollout = sample_rollout(train_bundle, initial_states=current_state)
        planned_states = rollout.states
        planned_actions = rollout.actions

        # Time steps for this rollout (offset by current step)
        rollout_time_steps = jnp.arange(horizon) + step
        action_time_steps = (
            jnp.arange(horizon - 1) + step
        )  # Actions are one step shorter

        # Add planned trajectory
        _add_planned_trajectory(
            fig,
            rollout_time_steps,
            action_time_steps,
            planned_states,
            planned_actions,
            state_dim,
            action_dim,
            total_plots,
            step,
        )

        # Take the first action and advance the state
        first_action = planned_actions[0]
        next_state = sys_dynamics(current_state, first_action)

        # Add actual trajectory
        _add_actual_trajectory(
            fig,
            step,
            current_state,
            next_state,
            first_action,
            state_dim,
            action_dim,
            total_plots,
        )

        # Update current state
        current_state = next_state

    # Finalize figure
    _finalize_figure(
        fig,
        state_dim,
        action_dim,
        total_plots,
        steps,
        "Receding Horizon Control (ProjectionLayer)",
    )

    # Log to wandb
    if wandb_step is not None:
        wandb.log({"receding_horizon_visualization": fig}, step=wandb_step)
    else:
        wandb.log({"receding_horizon_visualization": fig})

    print(
        f"Simulated {steps} steps with"
        f" {state_dim} states and {action_dim} actions"
    )
    print("Receding horizon visualization logged to wandb")

    # restore non deterministic training mode
    train_bundle.train()


def visualize_receding_horizon_mpc(
    mpc_layer,
    sys_dynamics,
    steps: int = 50,
    initial_state: jax.Array | None = None,
    wandb_step: int | None = None,
):
    """
    Visualize the receding horizon evolution using MPCLayer.
    At each step, show the full planned rollout and the actual step taken.
    """
    state_dim = sys_dynamics.state_dim
    action_dim = sys_dynamics.action_dim
    horizon = mpc_layer.mpc_problem.horizon

    # If no initial state is provided, use zero state
    if initial_state is None:
        initial_state = jnp.zeros(sys_dynamics.state_dim)

    # Create subplots
    fig, total_plots = _create_visualization_subplot(state_dim, action_dim)

    # Track the actual trajectory taken
    current_state = initial_state

    # Simulate receding horizon for the specified number of steps
    for step in range(steps):
        # Get rollout from current state using MPC
        rollout, mpc_info = mpc_layer(current_state)
        planned_states = rollout.states
        planned_actions = rollout.actions

        # Time steps for this rollout (offset by current step)
        rollout_time_steps = jnp.arange(horizon) + step
        action_time_steps = (
            jnp.arange(horizon - 1) + step
        )  # Actions are one step shorter

        # Add planned trajectory
        _add_planned_trajectory(
            fig,
            rollout_time_steps,
            action_time_steps,
            planned_states,
            planned_actions,
            state_dim,
            action_dim,
            total_plots,
            step,
            prefix="MPC ",
        )

        # Take the first action and advance the state
        first_action = planned_actions[0]
        next_state = sys_dynamics(current_state, first_action)

        # Add actual trajectory
        _add_actual_trajectory(
            fig,
            step,
            current_state,
            next_state,
            first_action,
            state_dim,
            action_dim,
            total_plots,
            prefix="MPC ",
        )

        # Update current state
        current_state = next_state

    # Finalize figure
    _finalize_figure(
        fig,
        state_dim,
        action_dim,
        total_plots,
        steps,
        "Receding Horizon Control (MPC Layer)",
    )

    # Log to wandb
    if wandb_step is not None:
        wandb.log({"receding_horizon_mpc_visualization": fig}, step=wandb_step)
    else:
        wandb.log({"receding_horizon_mpc_visualization": fig})

    print(
        f"MPC: Simulated {steps} steps with"
        f" {state_dim} states and {action_dim} actions"
    )
    print("MPC receding horizon visualization logged to wandb")


def visualize_comparison(
    train_bundle: TrainBundle,
    mpc_layer,
    steps: int = 50,
    initial_state: jax.Array | None = None,
    wandb_step: int | None = None,
):
    """
    Visualize and compare both ProjectionLayer and MPCLayer receding horizon.
    This function runs both visualizations for comparison.

    Args:
        train_bundle: TrainBundle containing the ProjectionLayer
        mpc_layer: MPCLayer for comparison
        steps: Number of simulation steps
        initial_state: Initial state for simulation
    """
    print("=== Visualizing ProjectionLayer ===")
    visualize_receding_horizon(train_bundle, steps, initial_state, wandb_step)

    print("\n=== Visualizing MPCLayer ===")
    sys_dynamics = train_bundle.system.sys_dynamics
    visualize_receding_horizon_mpc(
        mpc_layer, sys_dynamics, steps, initial_state, wandb_step
    )

    print("\n=== Comparison Complete ===")
    print("Both visualizations have been logged to wandb for comparison")
